import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        offset = input_dim - 3
        self.offset = offset
        self.conv1 = nn.Conv1d(input_dim, 128+offset, 1)
        self.conv2 = nn.Conv1d(128+offset, 128+offset, 1)
        self.conv3 = nn.Conv1d(128+offset, 256+offset, 1)
        self.conv4 = nn.Conv1d(256+offset, 512+offset, 1)
        self.bn1 = nn.BatchNorm1d(128+offset)
        self.bn2 = nn.BatchNorm1d(128+offset)
        self.bn3 = nn.BatchNorm1d(256+offset)
        self.bn4 = nn.BatchNorm1d(512+offset)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512+offset, 256+offset)
        self.fc2_m = nn.Linear(256+offset, 128+offset)
        self.fc3_m = nn.Linear(128+offset, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256+offset)
        self.fc_bn2_m = nn.BatchNorm1d(128+offset)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512+offset, 256+offset)
        self.fc2_v = nn.Linear(256+offset, 128+offset)
        self.fc3_v = nn.Linear(128+offset, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256+offset)
        self.fc_bn2_v = nn.BatchNorm1d(128+offset)

    def forward(self, x):
        # x: (B,N,3)
        x = x.transpose(1, 2) # (B,3,N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x)) # batch维度必须>1
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512+self.offset)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = (1 - betas).sqrt()
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]**2) / (1 - alpha_bars[i]**2)) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex) # 方差转标准差

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas



class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class PointwiseNet(nn.Module):

    def __init__(self, point_dim, context_dim, residual, return_sigma):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+3), # 这个3是time_emb,并非3维空间
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, point_dim, context_dim+3) # mu,log_sigma
        ])
        self.return_sigma = return_sigma
        if return_sigma:
            self.layers_v = ModuleList([
                ConcatSquashLinear(point_dim, 128, context_dim+3), # 这个3是time_emb,并非3维空间
                ConcatSquashLinear(128, 256, context_dim+3),
                ConcatSquashLinear(256, 512, context_dim+3),
                ConcatSquashLinear(512, 256, context_dim+3),
                ConcatSquashLinear(256, 128, context_dim+3),
                ConcatSquashLinear(128, point_dim, context_dim+3) # mu,log_sigma
            ])
    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        out_v = x
        if self.return_sigma:
            for i, (layer,layer_v) in enumerate(zip(self.layers,self.layers_v)):
                out = layer(ctx=ctx_emb, x=out)
                out_v = layer_v(ctx=ctx_emb, x=out_v)
                if i < len(self.layers) - 1:
                    out = self.act(out)
                    out_v = self.act(out_v)
            if self.residual:
                if self.return_sigma:
                    return x + out, out_v
                else:
                    return out
        else:
            for i, layer in enumerate(self.layers):
                out = layer(ctx=ctx_emb, x=out)
                if i < len(self.layers) - 1:
                    out = self.act(out)
            if self.residual:
                if self.return_sigma:
                    return x + out, out_v
                else:
                    return x + out
            else:
                return out


def cal_gamma(b_t,a_tm1,a_t,x_0,x_t1):
    gamma1 = (1-b_t).sqrt()/b_t * x_t1 + a_tm1/(1-a_tm1**2) * x_0
    gamma2 = (1-b_t)/(2*b_t) - 1/(2*(1-a_tm1**2))
    return gamma1, gamma2

def cal_kl_gauss(mu_1,sigma_1_sq,mu_2,sigma_2_sq):
    return ((mu_1-mu_2)**2)/(2*sigma_2_sq) # + sigma_1_sq/(2*sigma_2_sq) # - 0.5 * torch.log(sigma_1_sq)


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None, return_batch=False):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F). 须采样得到
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size) # (b,)
        alpha_bar_tm1 = self.var_sched.alpha_bars[[i-1 for i in t]] # (b,) x_0 -> x_t
        # alpha_bar_tm2 = self.var_sched.alpha_bars[[i-2 for i in t]] # x_0 -> x_t1
        beta_tm1 = self.var_sched.betas[[i-1 for i in t]] # (b,) x_t -> x_t1

        alpha_bar_tm1 = alpha_bar_tm1.view(-1,1,1)
        # alpha_bar_tm2 = alpha_bar_tm2.view(-1,1,1)
        # alpha_bar_t = alpha_bar_t.view(-1,1,1)
        beta_tm1 = beta_tm1.view(-1,1,1)
        
        # eps1,p1 = generate_WeightUniform_norm(x_0.shape[0],x_0.shape[1]*x_0.shape[2])
        # (B,N*d), (B)
        # eps1 = torch.reshape(eps1,x_0.shape)
        eps1 = torch.randn(x_0.shape)
        # 装进device
        eps1 = eps1.to(x_0.device)
        # p1 = p1.to(x_0.device)
        
        x_t = alpha_bar_tm1 * x_0 + (1-alpha_bar_tm1**2).sqrt() * eps1
        
        # mu_theta,log_sigma_theta = self.net(x_t, beta=beta_tm1, context=context)
        mu_theta = self.net(x_t, beta=beta_tm1, context=context)
        # sigma_theta_sq = (2*log_sigma_theta).exp()
        # # 1/alpha_bar_tm1 很大
        # mu_1 = 1/alpha_bar_tm1 * (x_t - beta_tm1 * mu_theta)
        # sigma_1_sq = beta_tm1**2 / alpha_bar_tm1**2 * sigma_theta_sq
        # 
        # gamma_1,gamma_2 = cal_gamma(beta_tm1, alpha_bar_tm2, alpha_bar_tm1, x_0, x_t)
        # 
        # mu_2 = - gamma_1 / (2*gamma_2)
        # sigma_2_sq = 1 / (2*gamma_2)
        # 
        # kl = cal_kl_gauss(mu_1,sigma_1_sq,mu_2,sigma_2_sq) # (B,N,3)
        # kl = kl.mean(-1).mean(-1)
        # loss = kl # * p1 # (B,)
        loss = ((mu_theta - eps1)**2).mean(-1).mean(-1)
        if return_batch:
            return loss
        else:
            return loss.sum()
        # if return_batch:
        #     loss = F.mse_loss(torch.flatten(e_theta,0,1), torch.flatten(e_rand,0,1), reduction='none')
        #     loss = loss.reshape((x_0.shape[0],x_0.shape[1],point_dim))
        # else:
        #     loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        # return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            #z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            #alpha = self.var_sched.alphas[t]
            #alpha_bar = self.var_sched.alpha_bars[t]
            #sigma = self.var_sched.get_sigmas(t, flexibility)
            #c0 = 1.0 / torch.sqrt(alpha)
            #c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            a_bar_tm1 = self.var_sched.alpha_bars[t]#[t-1]
            a_tm1 = self.var_sched.alphas[t]#[t-1]
            b_tm1 = self.var_sched.betas[t]#[t-1]
            x_t = traj[t]
            # e_theta = self.net(x_t, beta=beta, context=context)
            # mu_eps_theta,log_sigma_eps_theta = self.net(x_t, beta=b_tm1, context=context)
            mu_eps_theta = self.net(x_t, beta=b_tm1, context=context)
            eps_theta = mu_eps_theta
            # 
            sigma = self.var_sched.get_sigmas(t, flexibility)
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            c0 = 1 / a_tm1
            c1 = b_tm1 / (1-a_bar_tm1**2).sqrt()
            x_next = c0 * (x_t - c1 *eps_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        if ret_traj:
            return traj
        else:
            return traj[0]



# 为了训练 VAE, Mento Carlo 采样与权值生成
def generate_WeightUniform_norm(b,n):
    # e_rand是n元正态分布
    # pdf 在 b 维度上服从[0.5,1]的均匀分布
    # e_rand: (B,latent_dim)
    # pdf: (B)
    alpha = - 2 * np.log(torch.rand(b)*0.5+0.5)
    e_rand = torch.randn(b,n)
    alpha = alpha.unsqueeze(-1)
    e_rand = e_rand * (alpha / n) ** 0.5
    # log_normalization = (d / 2) * np.log(2 * np.pi)  # 归一化项
    log_exponent = -0.5 * (e_rand ** 2).sum(-1)  # 计算二次型
    pdf = log_exponent.exp()
    return e_rand, pdf

def get_time_embed(num_t):
    ts = torch.arange(0,num_t)
    ts = ts / num_t * torch.pi
    time_emb = torch.stack([torch.sin(ts), torch.cos(ts), ts],dim=1)  # (T, 3)
    return time_emb




class STAutoEncoder(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.use_freq = args.use_freq
        if args.use_freq:
            self.encoder = PointNetEncoder(zdim=args.latent_dim,input_dim=args.input_dim)
            self.diffusion = DiffusionPoint(
                net = PointwiseNet(point_dim=args.input_dim, context_dim=args.latent_dim, residual=args.residual, return_sigma=args.return_sigma),
                var_sched = VarianceSchedule(
                    num_steps=args.num_steps,
                    beta_1=args.beta_1,
                    beta_T=args.beta_T,
                    mode=args.sched_mode
                )
            )
        else:
            self.num_times = args.num_times
            self.encoder = STEncoder(num_times=args.num_times,dim_input=args.input_dim,dim_time=args.dim_z_time,dim_z=args.latent_dim)
            self.diffusion = DiffusionPoint(
                net = PointwiseNet(point_dim=args.input_dim, context_dim=args.latent_dim+3, residual=args.residual, return_sigma=args.return_sigma), # 包含了时间编码
                var_sched = VarianceSchedule(
                    num_steps=args.num_steps,
                    beta_1=args.beta_1,
                    beta_T=args.beta_T,
                    mode=args.sched_mode
                )
            )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B,T,N,D).
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, point_dim=self.input_dim, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, kl_weight=None):
        # x: (B,T,N,D) (not use_freq)
        # x: (B,N,2*T) (use_freq)
        m,log_sigma = self.encoder(x)
        # 重采样
        # e_rand,weight = generate_WeightUniform_norm(m.shape[0],m.shape[1])
        #e_rand = torch.randn((m.shape[0],m.shape[1])).to(m.device)
        #e_rand = e_rand.to(m.device)
        z = m # + log_sigma.exp() * e_rand # (b,latent_dim)
        loss = self.diffusion.get_loss(x, z, return_batch=True) #(B,)
        #weight = weight.to(x.device)
        #loss = (loss * weight).mean()
        loss = loss.mean()
        # 先验距离
        if kl_weight is not None:
            sigma_1_sq = (2*log_sigma).exp()
            kl = cal_kl_gauss(m,sigma_1_sq,0,1) # (b,latent_dim)
            kl = kl.mean(-1)
            # loss_kl = (kl * weight).mean()
            loss_kl = kl.mean()
            loss = loss + kl_weight * loss_kl
        return loss

    def get_st_encode(self,x,t,need_reparam=False):
        # x: (T,N,D)
        x = x.unsqueeze(0)
        self.eval()
        m,log_sigma = self.encoder(x)
        # 重采样
        if need_reparam:
            e_rand,weight = generate_WeightUniform_norm(m.shape[0],m.shape[1])
            e_rand = e_rand.to(m.device)
            z = m + log_sigma.exp() * e_rand
            m = z
        z = m
        # 加入时间编码
        z_t = self.add_t_to_z(z,t)
        return z_t
    
    def add_t_to_z(self,z,t):
        # z: (1,latent_dim)
        # 选择时间编码
        time_embed = get_time_embed(self.num_times)
        time_embed_select = time_embed[t,:]
        time_embed_select = time_embed_select.to(z.device)
        time_embed_select = time_embed_select.unsqueeze(0)
        # 拼接
        z_t = torch.concat([z,time_embed_select],dim=-1)
        return z_t
    
    def decode_all_time(self,z, num_points, flexibility=0.0, ret_traj=False):
        # z_t: (1,latent_dim)
        # x: (1,self.num_times,num_points,self.input_dim)
        x = torch.zeros((z.shape[0],self.num_times,num_points,self.input_dim)).to(z.device)
        for t in range(self.num_times):
            z_t = self.add_t_to_z(z,t)
            x_temp = self.decode(z_t,num_points,flexibility,ret_traj)
            # (1,num_points,3)
            x[:,t,:,:] = x_temp
        return x

