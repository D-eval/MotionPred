import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import sys

# from est import PREstimator
from motionTransformer import FixShapeEncoder, TransformerDecoder

from sgn import SGN

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


class PsyPredictor(nn.Module):
    def __init__(self,config_all):
        super().__init__()
        self.prediction_model = Transformer(config_all['prediction_model'])
        self.estimator = PREstimator(config_all['estimator'])
    def get_pred_train_params(self):
        # 获取 prediction_model, estimator.zl_embedding, estimator.post_enc 的参数给optim
        pred_params = list(self.prediction_model.parameters())
        pred_params += list(self.estimator.zl_embedding.parameters())
        pred_params += list(self.estimator.post_enc.parameters())
        return pred_params
    def get_est_train_params(self):
        # 获取 estimator.zs_embedding, estimator.prior_enc 的参数给optim
        est_params = list(self.estimator.zs_embedding.parameters())
        est_params += list(self.estimator.prior_enc.parameters())
        return est_params
    def freeze_pred_model(self):
        freeze_model(self.prediction_model)
        freeze_model(self.estimator.zl_embedding)
        freeze_model(self.estimator.post_enc)
        print('冻结预测模块')
    def freeze_est_model(self):
        freeze_model(self.estimator.zs_embedding)
        freeze_model(self.estimator.prior_enc)
        print('冻结估计模块')
    def unfreeze_pred_model(self):
        unfreeze_model(self.prediction_model)
        unfreeze_model(self.estimator.zl_embedding)
        unfreeze_model(self.estimator.post_enc)
        print('解冻预测模块')
    def unfreeze_est_model(self):
        unfreeze_model(self.estimator.zs_embedding)
        unfreeze_model(self.estimator.prior_enc)
        print('解冻估计模块')
    def get_loss_pred(self,inputs):
        # inputs: (b,t,n,d)
        b,t,n,d = inputs.shape
        time_long = self.estimator.len_long
        time_short = self.estimator.len_short
        if t < time_long:
            raise ValueError('t 必须比 {} 大'.format(time_long))
        inputs = inputs[:,:time_long,:,:]
        context = self.estimator.post_forward(inputs)
        if isinstance(context,tuple):
            mu, logsigma = context
            eps = torch.randn_like(mu).to(mu.device)
            sigma = logsigma.exp()
            z = mu + sigma * eps
            context = z
        pred_input = inputs[:,:-1,:,:] # (b,t_l-1,n,d)
        output, _ = self.prediction_model(pred_input,context)
        # (b,t_l,n,d)
        targets = inputs
        loss = self.prediction_model.get_loss(targets,output)
        return loss
    def get_loss_est(self,inputs):
        # inputs: (b,t,n,d)
        b,t,n,d = inputs.shape
        time_long = self.estimator.len_long
        time_short = self.estimator.len_short
        if t < time_long:
            raise ValueError('t 必须比 {} 大'.format(time_long))
        inputs = inputs[:,:time_long,:,:]
        loss = self.estimator.get_loss(inputs)
        return loss
    def get_valid_loss_pred(self,inputs):
        # inputs: (b,t,n,d)
        b,t,n,d = inputs.shape
        time_long = self.estimator.len_long
        time_short = self.estimator.len_short
        if t < time_long:
            raise ValueError('t 必须比 {} 大'.format(time_long))
        inputs = inputs[:,:time_long,:,:]
        context = self.estimator.sample_z(inputs[:,:time_short,:,:],given_l=False)
        pred_input = inputs[:,:-1,:,:]
        output, _ = self.prediction_model(pred_input,context)
        # (b,t_l,n,d)
        targets = inputs
        loss = self.prediction_model.get_loss(targets,output)
        return loss, (pred_input,targets)

'''
sgn_enc = SGN({
    'dim1':256,
    'seg':30,
    'num_joint':52,
    'joint_size':9,
    'bias':True,
    'batch_size':8,
    'train':True
})

x = torch.randn((8,30,52*9))
z = sgn_enc(x)
'''

'''
input = x
bs, step, dim = input.size()
num_joints = dim // sgn_enc.joint_size
input = input.view((bs, step, num_joints, sgn_enc.joint_size))
input = input.permute(0, 3, 2, 1).contiguous()
dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
pos = sgn_enc.joint_embed(input)

pos = sgn_enc.joint_embed.cnn[0](input)
'''