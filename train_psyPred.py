# 监视gpu使用
# watch -n 1 nvidia-smi
# train PsyPredictor
# 先练 pred
# 再练 est

# 采样若干个点,取loss最小的点

from psyPredictor import PsyPredictor

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import sys
sys.path.append('/home/vipuser/DL/Dataset100G/AMASS')
from read_data import AMASSDataset, collate_fn_min_seq

sys.path.append('/home/vipuser/DL/Dataset50G/Proj/Module')
from GIFWriter import draw_gif_t

import os
import time


save_dir = '/home/vipuser/DL/Dataset50G/save'
ckpt_name = 'psyPred_ckpt.pth'


device = torch.device('cuda')

# 加载数据集
train_set = AMASSDataset('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD', device)
train_set.apply_train_valid_divide()

valid_set = AMASSDataset('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD', device)
valid_set.apply_train_valid_divide(train=False)

train_loader = DataLoader(train_set, batch_size=8, collate_fn=collate_fn_min_seq, shuffle=True, drop_last=True)
valid_loader = DataLoader(train_set, batch_size=8, collate_fn=collate_fn_min_seq, shuffle=False, drop_last=True)

# 加载模型
config_pred = {
    'transformer_d_model':64,
    'num_joints':52,
    'transformer_num_heads_spacial':4,
    "transformer_num_heads_temporal": 4,
    'transformer_dropout_rate':0.1,
    'use_6d_outputs':False, # 6 or 9
    'transformer_window_length':121,
    'abs_pos_encoding':False,
    'transformer_num_layers':4,
    'shared_templ_kv':False,
    'temp_abs_pos_encoding':False,
    'temp_rel_pos_encoding':False,
    'residual_velocity':True,
    'loss_type':'geodesic'
}
config_pred['joint_size'] = 6 if config_pred['use_6d_outputs'] else 9

config_est = {
    'len_short':60,
    'len_long':120,
    'num_point':52,
    'input_dim':9,
    'embed_dim':512,
    'latent_dim':64,
    'layer_num':3,
    'use_post_sigma':True,
}

config_all = {
    'prediction_model':config_pred,
    'estimator':config_est
}

model = PsyPredictor(config_all)

# 优化器

def train_pred_one_epoch(model, train_loader, device='cuda', lr=1e-4):
    model.to(device)
    model.freeze_est_model()
    model.unfreeze_pred_model()
    params_for_train = model.get_pred_train_params()
    optimizer = torch.optim.Adam(params_for_train, lr=lr)
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        rot_mats, texts = batch
        optimizer.zero_grad()
        loss = model.get_loss_pred(rot_mats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        avg_loss = epoch_loss / num_batches
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    final_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Train Pred Loss: {final_avg_loss:.4f}")
    return final_avg_loss


def train_est_one_epoch(model, train_loader, device='cuda', lr=1e-4):
    model.to(device)
    model.freeze_pred_model()
    model.unfreeze_est_model()
    params_for_train = model.get_est_train_params()
    optimizer = torch.optim.Adam(params_for_train, lr=lr)
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        rot_mats, texts = batch
        optimizer.zero_grad()
        loss = model.get_loss_est(rot_mats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        avg_loss = epoch_loss / num_batches
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    final_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Train Est Loss: {final_avg_loss:.4f}")
    return final_avg_loss

visual_lst = [
    '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Female1Running_c3d/C5 - walk to run_poses.npz',
    '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Male1Walking_c3d/Walk B4 - Stand to Walk Back_poses.npz',
    '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Female1Gestures_c3d/D2 - Wait 1_poses.npz',
    '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Male1General_c3d/General A2 - Sway_poses.npz',
]

def valid_one_epoch(model, valid_loader):
    epoch_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc="Validing", leave=False)
        for batch in progress_bar:
            rot_mats, texts = batch
            loss, (pred,targets) = model.get_valid_loss_pred(rot_mats)
            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
        final_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Valid Loss: {final_avg_loss:.4f}")
        # if need_visual:
        #     visual_idx = 1
        #     text = texts[visual_idx]
        #     pred_rot_mats, _ = model.sample(rot_mats[visual_idx,:-prediction_step].unsqueeze(0),prediction_step)
        #     pred_cat = torch.cat([rot_mats[visual_idx,:-prediction_step].unsqueeze(0),pred_rot_mats],dim=1)
        #     ground_truth = rot_mats[visual_idx,:].unsqueeze(0)
        #     valid_set.processor.write_bvh(pred_cat[0],filename='pred_frame{}.bvh'.format(prediction_step))
        #     valid_set.processor.write_bvh(ground_truth[0],filename='real.bvh')
        #     xyz_pred = valid_set.processor.get_pos_tn3(pred_cat[0])
        #     xyz_real = valid_set.processor.get_pos_tn3(ground_truth[0])
        #     draw_gif_t(xyz_real.cpu().numpy(),xyz_pred.cpu().numpy(),train_set.processor.joint_to_parent,'compare_{}.gif'.format(text))
        #     print('可视化完成')
        return final_avg_loss


def get_model_size_in_gb(model):
    """
    计算模型所有参数的内存占用（单位：GB）
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()  # 总字节数
    total_gb = total_params / (1024 ** 3)  # 转换为 GB Byte->kB->MB->GB
    return total_gb

# model_size = get_model_size_in_gb(model)
# param_dict = dict(model.named_parameters())


if os.path.exists(os.path.join(save_dir,ckpt_name)):
    save_dict = torch.load(os.path.join(save_dir,ckpt_name), map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(save_dict['params'])
    print('加载参数成功')
else:
    save_dict = {'train_pred_loss':[],
                 'train_est_loss':[],
                'valid_loss':[],
                'params':None,
                'config':config_all}
    print('没有加载参数')

train_pred_loss_all = save_dict['train_pred_loss']
train_est_loss_all = save_dict['train_est_loss']
valid_loss_all = save_dict['valid_loss']




num_epoch = 20
for epoch in range(len(train_pred_loss_all),len(train_pred_loss_all)+num_epoch):
    print('epoch: ',epoch)
    start_time = time.time()
    train_loss = train_pred_one_epoch(model, train_loader)
    valid_loss = valid_one_epoch(model, valid_loader)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch} 用时: {epoch_time:.2f} 秒")
    train_pred_loss_all.append(train_loss)
    train_est_loss_all.append(None)
    valid_loss_all.append(valid_loss)
    save_dict['params'] = model.state_dict()
    torch.save(save_dict, os.path.join(save_dir,ckpt_name))
    print('保存成功')
    #plt.plot(train_pred_loss_all, label='pred loss')
    plt.plot(valid_loss_all, label='valid loss')
    #plt.plot(train_est_loss_all, label='est loss')
    plt.savefig(os.path.join(save_dir,'psy_loss.pdf'))
    plt.close()

num_epoch = 10
for epoch in range(len(train_pred_loss_all),len(train_pred_loss_all)+num_epoch):
    print('epoch: ',epoch)
    start_time = time.time()
    train_loss = train_est_one_epoch(model, train_loader)
    valid_loss = valid_one_epoch(model, valid_loader)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch} 用时: {epoch_time:.2f} 秒")
    train_pred_loss_all.append(None)
    train_est_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)
    save_dict['params'] = model.state_dict()
    torch.save(save_dict, os.path.join(save_dir,ckpt_name))
    print('保存成功')
    #plt.plot(train_pred_loss_all, label='pred loss')
    plt.plot(valid_loss_all, label='valid loss')
    #plt.plot(train_est_loss_all, label='est loss')
    plt.savefig(os.path.join(save_dir,'psy_loss.pdf'))
    plt.close()

# 画出 x_s 和 x_l 对应的两个分布 的 (降维后) 二维图

'''
for batch in valid_loader:
    rot_mats, texts = batch
    break

loss, (pred,targets) = model.get_valid_loss_pred(rot_mats)

inputs = rot_mats

b,t,n,d = inputs.shape
time_long = model.estimator.len_long
time_short = model.estimator.len_short

inputs = inputs[:,:time_long,:,:]
context = model.estimator.sample_z(inputs[:,:time_short,:,:],given_l=False)

btnd = inputs[:,:time_short,:,:]
mu_l, logsigma_l = model.estimator.prior_forward(btnd)
# 这里的 logsigma_l 太大

pred_input = inputs[:,:-1,:,:]
output, _ = model.prediction_model(pred_input,context)
# (b,t_l,n,d)
targets = inputs
loss = self.prediction_model.get_loss(targets,output)

btnd = inputs[:,:time_short,:,:]
mu_l, logsigma_l = model.estimator.prior_forward(btnd)

z = model.estimator.post_forward(inputs)

z_s = model.estimator.zs_embedding(btnd)

mu_s, logsigma_s = model.estimator.prior_enc(z_s)

rep = z_s

enc = model.estimator.prior_enc

rep = enc.first_linear(rep)
out = rep

i = 3

out = enc.linear_lst_1[i](out)
out = enc.batchnorm_lst_1[i](out)
out = F.relu(out)
out += rep
rep = out

for i in range(enc.layer_num):
    out = enc.linear_lst_1[i](out)
    out = enc.batchnorm_lst_1[i](out)
    out = F.relu(out)
    out += rep
    rep = out

'''
