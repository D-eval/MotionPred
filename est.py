# 短期表示 z_{short} 是 x[:t_1,:,:] 的表示 (D,)
# 长期表示 z_{long} 是 x[:,:,:]) 的表示 (D,)
# 建模 p(z_{long} | z_{short})

# z_{short} -> z -> z_{long}
# z \sim p(z | z_{short})
# z_{long} \sim p(z_{long} | z)

# infer
# mu_z, log_sigma_z = prior_enc(z_{short})
# sigma_z = log_sigma_z.exp()
# z = mu_z + rand * sigma_z
# z_{long} = dec(z)
# x_pred = model(x[:t1,:,:])
# loss = x_real - x_pred
# update z (只更新中间变量,不更新模型参数)
# z = argmin loss

# train
# z_{long} = x_to_z_long(x[:,:,:])
# z_{short} = x_to_z_short(x[:t1,:,:])
# mu_z_s, log_sigma_z_s = prior_enc(z_{short})
# mu_z_l, log_sigma_z_l = post_enc(z_{long})
# loss = cal_KL(mu_z_s, log_sigma_z_s, mu_z_l, log_sigma_z_l)

# 对比实验
# 基线模型 motionTransformer

# 消融实验
# z_{short/long}都可以通过cls token获得, 也可以通过这个展平后的mlp获得,


import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

# x -> z
# 灵活
class TNDEmbedding(nn.Module):
    def __init__(self,embed_dim,seq_len,num_point,input_dim):
        super().__init__()
        self.linear = nn.Linear(seq_len*num_point*input_dim, embed_dim)
    def forward(self,tnd):
        # tnd: (b,t,n,d)
        # return: (b,D)
        tnd_flattened = torch.flatten(tnd,1,-1)
        z = self.linear(tnd_flattened)
        return z

# z -> mu, log_sigma
class Encoder(nn.Module):
    def __init__(self,embed_dim,latent_dim,layer_num,use_sigma):
        super().__init__()
        self.layer_num = layer_num
        self.first_linear = nn.Linear(embed_dim, latent_dim)
        self.linear_lst_1 = nn.ModuleList([
            nn.Linear(latent_dim,latent_dim)
            for _ in range(layer_num)
        ])
        self.batchnorm_lst_1 = nn.ModuleList([
            nn.BatchNorm1d(latent_dim)
            for _ in range(layer_num)
        ])
        # 计算 mu
        self.linear_mu_lst = nn.ModuleList([
            nn.Linear(latent_dim,latent_dim)
            for _ in range(layer_num)
        ])
        self.batchnorm_mu_lst_1 = nn.ModuleList([
            nn.BatchNorm1d(latent_dim)
            for _ in range(layer_num)
        ])
        self.linear_mu_last = nn.Linear(latent_dim,latent_dim)
        # 计算 log_sigma
        self.use_sigma = use_sigma
        if use_sigma:
            self.linear_logsigma_lst = nn.ModuleList([
                nn.Linear(latent_dim,latent_dim)
                for _ in range(layer_num)
            ])
            self.batchnorm_logsigma_lst_1 = nn.ModuleList([
                nn.BatchNorm1d(latent_dim)
                for _ in range(layer_num)
            ])
            self.linear_logsigma_last = nn.Linear(latent_dim,latent_dim)
    def forward(self,rep):
        # rep: (b,embed_dim)
        rep = self.first_linear(rep)
        out = rep
        for i in range(self.layer_num):
            out = self.linear_lst_1[i](out)
            out = self.batchnorm_lst_1[i](out)
            out = F.relu(out)
            out += rep
            rep = out
        # cal mu
        mu = rep
        for i in range(self.layer_num):
            mu = self.linear_mu_lst[i](mu)
            mu = self.batchnorm_mu_lst_1[i](mu)
            mu = F.relu(mu)
            mu += rep
            rep = mu
        mu = self.linear_mu_last(mu)
        rep = out
        # cal log_sigma
        if self.use_sigma:
            log_sigma = rep
            for i in range(self.layer_num):
                log_sigma = self.linear_logsigma_lst[i](log_sigma)
                log_sigma = self.batchnorm_logsigma_lst_1[i](log_sigma)
                log_sigma = F.relu(log_sigma)
                log_sigma += rep
                rep = log_sigma
            log_sigma = self.linear_logsigma_last(log_sigma)
            # 限制 log_sigma
            log_sigma = 3.0 * torch.tanh(log_sigma / 3.0)
            return mu, log_sigma
        else:
            return mu


def cal_kl_divergence(mu1,mu2,log_sigma1,log_sigma2, eps=1e-8):
    # mu1: (b,D)
    b,dim = mu1.shape
    sigma1 = log_sigma1.exp()
    sigma2 = log_sigma2.exp()
    kl = 0.5 * ((log_sigma1.sum(1) - log_sigma2.sum(1)) - dim
    + ((mu1 - mu2)**2 / (sigma2**2 + eps)).sum(1) + (sigma1/sigma2).sum(1))
    # (b)
    return kl.mean()


def cal_max_likelihood_loss(mu,logsigma,x,eps=1e-8):
    # mu: (b,D)
    sigma = logsigma.exp()
    loss = logsigma.sum(1) + 0.5 * ((x-mu)**2 / (sigma**2+eps)).sum(1)
    loss = loss.mean()
    return loss


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


from motionTransformer import FixShapeEncoder

def get_model_cls(model_name):
    if model_name == 'FixShapeEncoder':
        return FixShapeEncoder
    

# 干脆不输入长序列, 只输入两个短序列吧
# tnd[:t,:,:] -> z_s
# tnd[t:2*t,:,:] -> z_l
class PREstimator(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        posterior_encoder_name = config['posterior_encoder_name']
        posterior_encoder_cls = get_model_cls(posterior_encoder_name)
        self.posterior_encoder = posterior_encoder_cls(config)
        
        prior_encoder_name = config['prior_encoder_name']
        prior_encoder_cls = get_model_cls(prior_encoder_name)
        self.prior_encoder = prior_encoder_cls(config)
        
        dim_decoder = config['dim_decoder']
        num_heads_decoder = config['num_heads_decoder']
        assert dim_decoder%num_heads_decoder==0
        depth_decoder = dim_decoder // num_heads_decoder
        C_encoder = depth_decoder * (depth_decoder - 1) / 2
        print('旋转基个数 {}'.format(C_encoder))
        
        len_short = config['len_short']
        len_long = config['len_long']
        num_point = config['num_point']
        input_dim = config['input_dim']
        embed_dim = config['embed_dim']
        latent_dim = config['latent_dim']
        layer_num = config['layer_num']
        use_post_sigma = config['use_post_sigma']
        

        
        self.use_posterior_sigma = config['use_posterior_sigma']
        self.use_prior_sigma = config['use_prior_sigma']
        
        self.len_short = len_short
        self.len_long = len_long
        self.num_point = num_point
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_post_sigma = use_post_sigma
    def forward(self,btnd):
        # btnd: (b,t,n,d)
        b,t,n,d = btnd.shape
        t_s = self.len_short
        t_l = self.len_long
        if ((n,d) != (self.num_point,self.input_dim)):
            raise ValueError('点数n 维数d 不符合, 期望 (n,d) 为 ({},{}), 得到了({},{})'.format(self.num_point,self.input_dim,n,d))
        if t < t_l:
            raise ValueError('时间长度t不足, 期望t大于 {}, 却得到了{}'.format(t_l,t))
        btnd_s = btnd[:,:t_s,:,:]
        btnd_l = btnd[:,:t_l,:,:]
        mu_s, logsigma_s = self.prior_forward(btnd_s)
        if self.use_post_sigma:
            mu_l, logsigma_l = self.post_forward(btnd_l)
            return mu_s, logsigma_s, mu_l, logsigma_l
        else:
            z = self.post_forward(btnd_l)
            return mu_s, logsigma_s, z
    def prior_forward(self,btnd):
        # btnd: (b,t_s,n,d)
        b,t,n,d = btnd.shape
        if t != self.len_short:
            raise ValueError('t 必须为 {}'.format(self.len_short))
        z_s = self.zs_embedding(btnd)
        mu_s, logsigma_s = self.prior_enc(z_s)
        return mu_s, logsigma_s
    def post_forward(self,btnd):
        # btnd: (b,t_l,n,d)
        b,t,n,d = btnd.shape
        if t != self.len_long:
            raise ValueError('t 必须为 {}'.format(self.len_long))
        z_l = self.zl_embedding(btnd)
        if self.use_post_sigma:
            mu_l, logsigma_l = self.post_enc(z_l)
            return mu_l, logsigma_l
        else:
            z = self.post_enc(z_l)
            return z
    def get_loss(self,btnd):
        # btnd: (b,t,n,d)
        if self.use_post_sigma:
            mu_s, logsigma_s, mu_l, logsigma_l = self(btnd)
            loss = cal_kl_divergence(mu_l,mu_s,logsigma_l,logsigma_s)
            return loss
        else:
            mu_s, logsigma_s, z = self(btnd) # (b,D)
            loss = cal_max_likelihood_loss(mu_s, logsigma_s, z)
            return loss            
    def sample_z(self,btnd,eps=None,given_l=True):
        # btnd: (b,t_s,n,d)
        # return z: (b,D)
        b,t,n,d = btnd.shape
        if t != self.len_short:
            raise ValueError('t 必须为 {}'.format(self.len_short))
        if given_l:
            if self.use_post_sigma:
                mu_l, logsigma_l = self.post_forward(btnd)
            else:
                # 决定情况
                z = self.post_forward(btnd)
                return z
        else:
            mu_l, logsigma_l = self.prior_forward(btnd)
        if eps is None:
            eps = torch.randn_like(mu_l).to(mu_l.device)
        sigma_l = logsigma_l.exp()
        z = mu_l + sigma_l * eps
        return z
    def optim_z(self,btnd_s,btnd_l,pred_model,loss_fc,num_epoch=100,lr=1e-2):
        # btnd_s, btnd_l: (b,t_s,n,d), 时间连续
        # pred_model: (b,t_s+1,n,d)->(b,t_s,n,d)
        # 须先使用 freeze_model
        # loss_fc: (b,t_s,n,d)*2 -> float
        b,t,n,d = btnd_s.shape
        # eps = argmin loss
        # 迭代num_epoch次
        device = btnd_s.device
        eps = torch.randn(b, self.latent_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([eps], lr=lr)
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            z = self.sample_z(btnd_s,eps=eps,given_l=False)
            # (b,D)
            btnd_l_pred = pred_model(btnd_s,context=z)
            loss = loss_fc(btnd_l,btnd_l_pred)
            # 用 loss 更新 eps
            loss.backward()
            optimizer.step()
            # 可选：打印监控
            if epoch % 10 == 0 or epoch == num_epoch - 1:
                print(f"[optim_z] Epoch {epoch}/{num_epoch} - Loss: {loss.item():.6f}")
        # 返回时将 eps 从计算图中分离
        with torch.no_grad():
            z_final = self.sample_z(btnd_s, eps=eps, given_l=False)
        # 找到 z_final 后, 再更新 pred_model
        return eps.detach(), z_final.detach(), loss.item()

'''
# 错误的layernorm
class Layernorm(nn.Module):
    def __init__(self,d_model,eps=1e-8):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(d_model))
        self.sigma = nn.Parameter(torch.randn(d_model))
        self.eps = 1e-8
    def forward(self,x):
        # x: (B,T,D)
        B,T,D = x.shape
        y = []
        for b in range(B):
            y += [ (x[b] - self.mu.unsqueeze(0)) / (self.sigma.unsqueeze(0) + self.eps) ]
        return torch.stack(y,dim=0)
'''

# 正确的layernorm
class Layernorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # scale
        self.beta = nn.Parameter(torch.zeros(d_model))  # shift
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D)
        mean = x.mean(dim=-1, keepdim=True)        # (B, T, 1)
        std = x.std(dim=-1, keepdim=True)          # (B, T, 1)
        normed = (x - mean) / (std + self.eps)      # 标准化
        out = self.gamma * normed + self.beta       # 仿射变换
        return out

