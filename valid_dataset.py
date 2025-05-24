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

dataset_name = "KIT"
save_dir = '/home/vipuser/DL/Dataset50G/save'
ckpt_name = 'ckpt_prior_mt_diffusion.pth'
dataset_dir = '/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/{}'.format(dataset_name)
model_config_file = "TransDiff_usehand.json"
device = torch.device('cuda')
# 加载验证集
valid_set = AMASSDataset(dataset_dir, device, time_len=10, use_hand=True, target_fps=30, use_6d=True)
valid_set.apply_train_valid_divide(train=False)

idx = -20

num_joints = valid_set.num_joints
npz_path, text, start_idx = valid_set.samples[idx]
bdata = np.load(npz_path)
full_pose = torch.Tensor(bdata['poses']) # (T,N*3)

fps = int(bdata['mocap_framerate'].item())
target_fps = valid_set.target_fps

# 下采样
from read_data import down_sample
full_pose = down_sample(full_pose, fps, target_fps)

rot_mats = valid_set.processor.fullpose_to_RotMat(full_pose) # (T,N,3,3)
if valid_set.use_6d:
    rot_mats_6d = rot_mats[:,:,:2,:] # (T,N,3,2)
rot_mats_6d = torch.flatten(rot_mats_6d,-2,-1)



T,N,_ = rot_mats_6d.shape
rot_mat_9d = torch.flatten(rot_mats_6d,0,1) # (?,6)
rot_mat_9d = compute_rotation_matrix_from_ortho6d(rot_mat_9d)
rot_mat_9d = torch.reshape(rot_mat_9d, (T,N,9))
valid_set.processor.write_bvh(rot_mat_9d,filename='{}_{}.bvh'.format(dataset_name,name))


for i in range(52):
    print(rot_mats[:,i,0].std())

for i in range(52):
    print(rot_mats[:,i,0].mean())

