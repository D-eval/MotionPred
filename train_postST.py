# 监视gpu使用
# watch -n 1 nvidia-smi
# train motionTransformer

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from motionTransformer import TransformerAutoEncoder
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import sys
sys.path.append('/home/vipuser/DL/Dataset100G/AMASS')
from read_data import AMASSDataset, collate_fn_min_seq, report_dataset

sys.path.append('/home/vipuser/DL/Dataset50G/Proj/Module')
from GIFWriter import draw_gif_t

import os
import time


save_dir = '/home/vipuser/DL/Dataset50G/save'
ckpt_name = 'ckpt_postST.pth'
dataset_name = 'CMU'
dataset_dir = '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU'

time_len = 10
use_hand = False
target_fps = 30

batch_size = 8

device = torch.device('cuda')

# 加载模型
config = {
    'transformer_d_model':64,
    'num_joints':52,
    'transformer_num_heads_spacial':4,
    "transformer_num_heads_temporal": 4,
    'transformer_dropout_rate':0.1,
    'use_6d_outputs':False, # 6 or 9
    'transformer_window_length':121, # 120 + 1
    'abs_pos_encoding':False,
    'transformer_num_layers':4,
    'shared_templ_kv':False,
    'temp_abs_pos_encoding':False,
    'temp_rel_pos_encoding':False,
    'residual_velocity':True,
    'loss_type':'geodesic'
}


report_dataset(dataset_dir, dataset_name)


# 加载数据集
train_set = AMASSDataset(dataset_dir, device, time_len, use_hand, target_fps)
train_set.apply_train_valid_divide()

valid_set = AMASSDataset(dataset_dir, device, time_len, use_hand, target_fps)
valid_set.apply_train_valid_divide(train=False)

train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn_min_seq, shuffle=True)
valid_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn_min_seq, shuffle=False)


config['joint_size'] = 6 if config['use_6d_outputs'] else 9

model = Transformer(config)


# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



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



def valid_one_epoch(model, valid_loader, valid_set, need_visual=True):
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
    save_dict = {'train_loss':[],
                'valid_loss':[],
                'params':None,
                'config':config}
    print('没有加载参数')

train_loss_all = save_dict['train_loss']
valid_loss_all = save_dict['valid_loss']

num_epoch = 50
for epoch in range(len(train_loss_all),len(train_loss_all)+num_epoch):
    print('epoch: ',epoch)
    start_time = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer)
    valid_loss = valid_one_epoch(model, valid_loader, valid_set, need_visual=(epoch % 10 == 0))
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch} 用时: {epoch_time:.2f} 秒")
    train_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)
    save_dict['params'] = model.state_dict()
    torch.save(save_dict, os.path.join(save_dir,ckpt_name))
    print('保存成功')

    plt.plot(train_loss_all)
    plt.savefig(os.path.join(save_dir,'loss_compared.pdf'))
    plt.close()

