# 监视gpu使用
# watch -n 1 nvidia-smi
# train motionTransformer

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from motionTransformerDiffusion import TransformerDiffusion, compute_rotation_matrix_from_ortho6d
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

# 向后预测长动作评估
def get_multi_time_pred_loss(idx, pred_step=150, need_bvh=True):
    # 整个句子10 s 预测 5 s
    all_frame_loss = []
    rot_mat = valid_set[idx][0].to(device) # (T, N, 6)
    text = valid_set[idx][1]
    T = rot_mat.shape[0]
    T_history = T - pred_step

    rot_mat = rot_mat.unsqueeze(0) # (1, T, N, 6)
    history_poses = rot_mat[:, :T_history] # (1, T_h, N, 6)

    model.eval()
    with torch.no_grad():
        context = model.encode(rot_mat) # (1, C, D)
        progress_bar = tqdm(range(pred_step),desc="Sampling")
        for t in progress_bar:
            new_pose = model.sample(history_poses, context) # # (1, 1, N, d_in)
            history_poses = torch.cat([history_poses, new_pose], dim=1)
            frame_loss = ((rot_mat[:,T_history+t] - new_pose[:,0])**2).mean()
            all_frame_loss.append(frame_loss.item())
            progress_bar.set_postfix(loss="{:.4f}".format(frame_loss.item()))
        # history_poses (1, T, N, 6)
    if not need_bvh:
        return all_frame_loss
    _, T, N, _ = history_poses.shape
    output_rot_mat = torch.flatten(history_poses,0,2) # (?,6)
    output_rot_mat = compute_rotation_matrix_from_ortho6d(output_rot_mat)
    output_rot_mat = torch.reshape(output_rot_mat, (T,N,9))
    valid_set.processor.write_bvh(output_rot_mat,filename='posterior_diffusion_frame_{}_motion_{}.bvh'.format(pred_step,text[:-4]))
    print('成功保存bvh文件')
    fps = valid_set.target_fps
    interval = 1/fps
    x_times = np.arange(0,len(all_frame_loss),interval)
    plt.plot(x_times,all_frame_loss)
    plt.xlabel('time (seconds)')
    plt.ylabel('MSE loss by diffusion')
    plt.title('MSE per frame, action:{}'.format(text[:-4]))
    plt.tight_layout()
    save_path = os.path.join(save_dir, "posterior_diffusion_pred_{}_steps_{}.pdf".format(pred_step, text[:-4]))
    plt.savefig(save_path)
    plt.close()
    print('保存多步预测MSE变化至 {}'.format(save_path))
    return all_frame_loss

def get_single_time_pred_loss(idx):
    rot_mat = valid_set[idx][0].to(device) # (T, N, 6)
    text = valid_set[idx][1]
    T = rot_mat.shape[0]
    rot_mat = rot_mat.unsqueeze(0) # (1, T, N, 6)
    history_poses = rot_mat[:, :-1] # (1, T_h, N, 6)
    target_poses = rot_mat[:, 1:] # (1, T_h, N, 6)
    model.eval()
    with torch.no_grad():
        context = model.encode(rot_mat) # (1, C, D)
        pred_poses = model.sample_each(history_poses, context)
        loss = ((target_poses-pred_poses)**2).mean()
    return loss.item()

# 计算整个验证集上的valid_loss
# 向后预测一步，完整去噪
# 评估最终loss
def valid_all():
    valid_set_len = len(valid_set)
    all_idx = range(valid_set_len)
    progress_bar = tqdm(all_idx,desc="Validing")
    sum_loss = 0
    for idx in progress_bar:
        temp_loss = get_single_time_pred_loss(idx)
        sum_loss += temp_loss
        progress_bar.set_postfix(loss="{:.4f}".format(temp_loss))
    avg_loss = sum_loss / valid_set_len
    print('Valid loss: {:.4f}'.format(avg_loss))
    return avg_loss


save_dir = '/home/vipuser/DL/Dataset50G/save'
ckpt_name = 'ckpt_mt.pth'
dataset_dir = '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD'
model_config_file = "Trans.json"
device = torch.device('cuda')
# 加载验证集
valid_set = AMASSDataset(dataset_dir, device, time_len=10, use_hand=True, target_fps=30, use_6d=True)
valid_set.apply_train_valid_divide(train=False)
# 读取模型配置
with open(model_config_file, "r") as f:
    config = json.load(f)
# 加载模型
model = TransformerDiffusion(**config)
print('后验信息长度:{}'.format(model.encoder.pool.query.shape[0]))
print('输入序列长度:{}'.format(valid_set[0][0].shape[0]))
# assert valid_set[0][0].shape[0] > 2 * model.encoder.pool.query.shape[0],"后验信息过多"
# 加载模型参数
if os.path.exists(os.path.join(save_dir,ckpt_name)):
    save_dict = torch.load(os.path.join(save_dir,ckpt_name), map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(save_dict['params'])
    print('加载参数成功')
else:
    save_dict = {'train_loss':[],
                'valid_loss':[],
                'params':None,
                'config':config}
    print('没有加载参数')

model.to(device)
# 整个句子10 s 预测 5 s
idx=0
pred_step= 50 # 150
need_bvh=True

all_frame_loss = []
rot_mat = valid_set[idx][0].to(device) # (T, N, 6)
text = valid_set[idx][1]
T = rot_mat.shape[0]
T_history = T - pred_step

rot_mat = rot_mat.unsqueeze(0) # (1, T, N, 6)
history_poses = rot_mat[:, :T_history] # (1, T_h, N, 6)

model.eval()
with torch.no_grad():
    context = model.encode(rot_mat) # (1, C, D)
    progress_bar = tqdm(range(pred_step),desc="Sampling")
    for t in progress_bar:
        new_pose = model.sample(history_poses, context) # # (1, 1, N, d_in)
        history_poses = torch.cat([history_poses, new_pose], dim=1)
        frame_loss = ((rot_mat[:,T_history+t] - new_pose[:,0])**2).mean()
        all_frame_loss.append(frame_loss.item())
        progress_bar.set_postfix(loss="{:.4f}".format(frame_loss.item()))
    # history_poses (1, T, N, 6)
_, T, N, _ = history_poses.shape
output_rot_mat = torch.flatten(history_poses,0,2) # (?,6)
output_rot_mat = compute_rotation_matrix_from_ortho6d(output_rot_mat)
output_rot_mat = torch.reshape(output_rot_mat, (T,N,9))
valid_set.processor.write_bvh(output_rot_mat,filename='prior_diffusion_frame_{}_motion_{}.bvh'.format(pred_step,text[:-4]))
print('成功保存bvh文件')
fps = valid_set.target_fps
interval = 1/fps
x_times = np.arange(0,len(all_frame_loss)/fps,interval)
plt.plot(x_times,all_frame_loss)
plt.xlabel('time (seconds)')
plt.ylabel('MSE loss by diffusion')
plt.title('MSE per frame, action:{}'.format(text[:-4]))
plt.tight_layout()
save_path = os.path.join(save_dir, "posterior_diffusion_pred_{}_steps_{}.pdf".format(pred_step, text[:-4]))
plt.savefig(save_path)
plt.close()
print('保存多步预测MSE变化至 {}'.format(save_path))

all_times = np.arange(0,output_rot_mat.shape[0]) * interval
for idx in range(output_rot_mat.shape[1]):
    plt.plot(all_times, output_rot_mat[:,idx,0].cpu(), label='joint {}'.format(idx))

plt.xlabel('time (seconds)')
plt.ylabel('angle')
save_path_motion = os.path.join(save_dir, "posterior_diffusion_pred_{}_steps_{}_motion.pdf".format(pred_step, text[:-4]))
plt.savefig(save_path_motion)
plt.close()


'''
inputs = rot_mat # (1, T, N, d_in)
context = context
tau = sample_tau(B, T, Tau, inputs.device)

beta_tau = self.diffusion_schedule.betas[tau] # (B, T)
alpha_tau = self.diffusion_schedule.alphas[tau] # (B, T)
alpha_bar_tau = self.diffusion_schedule.alpha_bars[tau] # (B, T)
v = inputs[:, 1:, :, :] - inputs[:, :-1, :, :] # (B, T, N, d_in)
B, T, N, D = v.shape
epsilon = torch.randn_like(v) # (B, T, N, d_in)
# 加噪
sqrt_alpha_bar = alpha_bar_tau.sqrt().view(B, T, 1, 1)
sqrt_one_minus_alpha_bar = (1 - alpha_bar_tau).sqrt().view(B, T, 1, 1)
# 缩放到同一水平
epsilon_scaled = epsilon * self.noise_scale
v_tau = v * sqrt_alpha_bar + epsilon_scaled * sqrt_one_minus_alpha_bar  # (B, T, N, D)

# tau: (B, T)
tau_embed = self.tau_embed(tau) # (B, T, D)

history_poses = inputs[:, :-1, :, :] # (B, T, N, d_in)
epsilon_pred_scaled = self.decode(v_tau, history_poses, tau_embed, context, None) # (B, T, N, d_in)
epsilon_pred = epsilon_pred_scaled / self.noise_scale


'''