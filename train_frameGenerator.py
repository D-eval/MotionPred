# 监视gpu使用
# watch -n 1 nvidia-smi
# train motionTransformer

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from motionTransformerDiffusionMultiStep import FrameRepNet, compute_rotation_matrix_from_ortho6d
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from valid_mt_multi_diffusion import visual_HifiDiff_result

# 用
import subprocess
def push_loss_curve_to_github(files, help='Update loss curve'):
    # push_loss_curve_to_github(['curve.pdf', 'loss.txt'])
    try:
        if isinstance(files, str):
            files = [files]
        subprocess.run(['git', 'add'] + files, check=True)
        subprocess.run(['git', 'commit', '-m', help], check=True)
        subprocess.run(['git', 'push'], check=True)
        print(f"[Git] {files} pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"[Git Error] {e}")


import sys
from read_data import AMASSDataset, collate_fn_min_seq
from Module.GIFWriter import draw_gif_t

import os
import time



# 训练函数
def train_one_epoch(model, train_loader, optimizer, device='cuda'):
    model.to(device)
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        rot_mats, texts = batch
        optimizer.zero_grad()
        loss = model.get_loss(rot_mats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
        avg_loss = epoch_loss / num_batches
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
    final_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Train Loss: {final_avg_loss:.4f}")
    return final_avg_loss


# 验证函数
def valid_one_epoch(model, valid_loader, valid_set, need_visual=False):
    prediction_step = 60
    epoch_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc="Validing", leave=False)
        for batch in progress_bar:
            rot_mats, texts = batch
            loss = model.get_loss(rot_mats)
            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
        final_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Valid Loss: {final_avg_loss:.4f}")
        if need_visual:
            visual_idx = 1
            text = texts[visual_idx]
            pred_rot_mats, _ = model.sample(rot_mats[visual_idx,:-prediction_step].unsqueeze(0),prediction_step)
            pred_cat = torch.cat([rot_mats[visual_idx,:-prediction_step].unsqueeze(0),pred_rot_mats],dim=1)
            ground_truth = rot_mats[visual_idx,:].unsqueeze(0)
            valid_set.processor.write_bvh(pred_cat[0],filename='pred_frame{}.bvh'.format(prediction_step))
            valid_set.processor.write_bvh(ground_truth[0],filename='real.bvh')
            xyz_pred = valid_set.processor.get_pos_tn3(pred_cat[0])
            xyz_real = valid_set.processor.get_pos_tn3(ground_truth[0])
            draw_gif_t(xyz_real.cpu().numpy(),xyz_pred.cpu().numpy(),train_set.processor.joint_to_parent,'compare_{}.gif'.format(text))
            print('可视化完成')
        return final_avg_loss

# 获取模型大小
def get_model_size_in_gb(model):
    """
    计算模型所有参数的内存占用（单位：GB）
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()  # 总字节数
    total_gb = total_params / (1024 ** 3)  # 转换为 GB Byte->kB->MB->GB
    return total_gb

def init_model(model,save_path,config):
    # 加载模型参数
    if os.path.exists(save_path):
        save_dict = torch.load(save_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(save_dict['params'], strict=False)
        print('加载参数成功')
    else:
        save_dict = {'train_loss':[],
                    'valid_loss':[],
                    'params':None,
                    'config':config}
        print('没有加载参数')
    return save_dict



save_dir = '/home/vipuser/DL/Dataset50G/save'
ckpt_name = 'ckpt_HifiDiff.pth' #'ckpt_multi_diffusion.pth'
dataset_dir = ['/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/KIT',
               '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU',
               '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/BMLmovi',
               '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD',
               '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/Eyes_Japan_Dataset'] # 去AMASS官网下载数据集
model_config_file = "T_frame_rep_net_attn.json"
device = torch.device('cuda')
# 加载数据集
train_set = AMASSDataset(dataset_dir, device, time_len=10, use_hand=True, target_fps=30, use_6d=True)
train_set.apply_train_valid_divide()
# 加载验证集
valid_set = AMASSDataset(dataset_dir, device, time_len=10, use_hand=True, target_fps=30, use_6d=True)
valid_set.apply_train_valid_divide(train=False)
# 加载数据加载器
train_loader = DataLoader(train_set, batch_size=16, collate_fn=collate_fn_min_seq, shuffle=True)
valid_loader = DataLoader(train_set, batch_size=16, collate_fn=collate_fn_min_seq, shuffle=False)
# 读取模型配置
with open(model_config_file, "r") as f:
    config = json.load(f)
# 加载模型
model = FrameRepNet(**config)
print('后验信息长度:{}'.format(model.vae.latent_dim))
print('输入序列长度:{}'.format(train_set[0][0].shape[0]))
# assert train_set[0][0].shape[0] > 2 * model.encoder.pool.query.shape[0],"后验信息过多"

# 加载模型优化器
# pretrain后冻结encoder
params_to_train = list(model.parameters()) #.get_train_params_without_encoder()
optimizer = torch.optim.Adam(params_to_train, lr=1e-4)


# 报告模型大小
model_size = get_model_size_in_gb(model)
print('模型大小为: {:.3f} GB'.format(model_size))
# 加载历史训练数据
save_dict_path = os.path.join(save_dir,ckpt_name)
save_dict = init_model(model,save_dict_path,config)
# 获取损失函数列表
train_loss_all = save_dict['train_loss']
valid_loss_all = save_dict['valid_loss']
print("训练空间扩散模型")
num_epoch = 20
for epoch in range(len(train_loss_all),len(train_loss_all)+num_epoch):
    train_loader = DataLoader(train_set, batch_size=12, collate_fn=collate_fn_min_seq, shuffle=True) # 每个epoch shuffle一下
    print('epoch: ',epoch)
    start_time = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer)
    # valid_loss = valid_one_epoch(model, valid_loader, valid_set, need_visual=False)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch} 用时: {epoch_time:.2f} 秒")
    train_loss_all.append(train_loss)
    # valid_loss_all.append(valid_loss)
    save_dict['params'] = model.state_dict()
    torch.save(save_dict, os.path.join(save_dir,ckpt_name))
    print('保存成功')
    plt.plot(train_loss_all)
    plt.savefig(os.path.join(save_dir,'loss_{}.pdf'.format(ckpt_name[:-4])))
    plt.close()
    visual_idx = visual_HifiDiff_result(valid_set=train_set, model=model, name=ckpt_name[:-4], idx=None)
    print('结果可视化完成')
    # 实时推送github，方便手机端远程查看
    with open('./loss.txt','a') as f:
        f.write('epoch:{} loss: {}\n'.format(epoch,train_loss))
    push_loss_curve_to_github(['curve.png','curve_real.png','loss.txt','README.md'])



'''
idx = 0
rot_mat = valid_set[idx][0].to(device) # (T, N, 6)
text = valid_set[idx][1]
T = rot_mat.shape[0]
rot_mat = rot_mat.unsqueeze(0) # (1, T, N, 6)
model.eval()
with torch.no_grad():
    context = model.encode(rot_mat) # # (1, T, D)

poses = rot_mat
B,T,N,d = poses.shape
D = model.d_out
poses = torch.flatten(poses,0,1) # (B*T,N,d)
poses_rep_seq = model.vae.encode(poses) # (B*T,N,D)
poses = torch.reshape(poses_rep_seq, (B,T,D))
'''



'''
progress_bar = tqdm(range(20))
model.to(device)
for i in progress_bar:
    visual_idx = visual_result(valid_set=train_set, model=model, idx=None, disp_dir='./output')
'''

'''
for batch in train_loader:
    rot_mats, texts = batch
    break
'''

'''
rot_mat = train_set[0][0]
v = rot_mat[1:] - rot_mat[:-1] # 1e-5

mu = 0
for i in range(1000):
    rot_mat = train_set[i][0]
    v = rot_mat[1:] - rot_mat[:-1] # 1e-5
    mu += v.mean()
'''

'''
train_iter = iter(train_loader)
batch = next(train_iter)
rot_mats, texts = batch
normed_input = normer.norm(rot_mats)
mu1 = normer.mu

normed_input = normer.norm(train_set[1][0].unsqueeze(0))
mu2 = normer.mu
'''