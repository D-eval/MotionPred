
'''
帮我写一个Dataset,读取npz文件,目录结构如下
ACCAD
|-Female1General_c3d
|   |-D1 Urban 1_poses.npz
|   |-D2 Wait 1_poses.npz
|   ...
|-Female1Gestures_c3d
|   |-D1 Urban 1_poses.npz
|   |-D2 Wait 1_poses.npz
|   ...
|-Female1Running_c3d
|   |-C2 Run to stand_poses.npz
|   ...
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import sys
# sys.path.append('/home/vipuser/DL/Dataset50G/Proj/Module')
from Module.BVHWriter import BVHWriter
from Module.OBJWriter import write_obj

body_model_path = '/home/vipuser/DL/Dataset100G/AMASS/Code/human_body_prior/src/human_body_prior/body_model'
# 去smpl官网下载body_model, 替换为你的路径
sys.path.append(body_model_path)
from lbs import batch_rodrigues, blend_shapes, vertices2joints, batch_rigid_transform

from scipy.spatial.transform import Rotation as R

import pickle
import random

# 替换为你的保存目录
save_dir = '/home/vipuser/DL/Dataset50G/save'

class Processor:
    def __init__(self,device,use_hand=True):
        # device = torch.device('cuda')
        gender_2_smpl_dict = {}
        support_dir = '/home/vipuser/DL/Dataset100G/AMASS/Code/amass/support_data'
        for subject_gender in ['female','male','neutral']:
            bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
            smpl_dict = np.load(bm_fname, encoding='latin1')
            gender_2_smpl_dict[subject_gender] = {}
            temp_dict = gender_2_smpl_dict[subject_gender]
            J_regressor = torch.Tensor(smpl_dict['J_regressor']).to(device) # 6890->52
            v_template = torch.Tensor(smpl_dict['v_template']).to(device) # (6890,3)
            shapedirs = torch.Tensor(smpl_dict['shapedirs'][:, :, :16]).to(device)
            temp_dict['J_regressor'] = J_regressor
            temp_dict['v_template'] = v_template
            temp_dict['shapedirs'] = shapedirs

        num_joints = smpl_dict['kintree_table'].shape[1] if use_hand else 21
        parents = smpl_dict['kintree_table'][0]
        if use_hand:
            joint_to_parent = {smpl_dict['kintree_table'][1][i]:smpl_dict['kintree_table'][0][i] for i in range(num_joints)}
        else:
            joint_to_parent = {smpl_dict['kintree_table'][1][i]:smpl_dict['kintree_table'][0][i] for i in range(num_joints)}
        joint_to_parent[0] = None
        writer = BVHWriter(joint_to_parent)
        
        self.device = device
        self.gender_2_smpl_dict = gender_2_smpl_dict
        self.num_joints = num_joints
        self.parents = parents
        self.joint_to_parent = joint_to_parent
        self.writer = writer

    def fullpose_to_RotMat(self,full_pose):
        # full_pose: (T,num_joints * 3) 轴角制
        # return: (T,num_joints,9) 展平的旋转矩阵
        flatten_pose = full_pose.view(-1, 3) # (T*num_joints,3)
        rot_mats = batch_rodrigues(
            flatten_pose, dtype=torch.float32).view([full_pose.shape[0], -1, 3, 3])
        rot_mats = torch.Tensor(rot_mats).to(self.device)
        return rot_mats
    
    def write_bvh(self,rot_mats,root_motion=None,bdata=None,filename='real.bvh',save_dir=None):
        save_dir = "/home/vipuser/DL/Dataset50G/save" if save_dir is None else save_dir
        # bdata: smplx .npz data
        # rot_mats: (t,52,9) or (t,21,9)
        # filename: path/to/yourfile.bvh
        use_rot_mats = (rot_mats is not None)
        if rot_mats is None:
            if bdata is None:
                raise ValueError('rot_mats和bdata必须有其中一个')
            full_pose = torch.Tensor(bdata['poses'])# [:,3:])
            flatten_pose = full_pose.view(-1, 3)
            rot_mats = batch_rodrigues(
                flatten_pose, dtype=torch.float32).view([full_pose.shape[0], -1, 9])
            rot_mats = torch.Tensor(rot_mats).to(self.device)
        
        times = rot_mats.shape[0]
        
        if root_motion is None:
            # root_motion = bdata['trans']
            if not use_rot_mats:
                root_motion = bdata['trans']
            else:
                root_motion = torch.zeros((times,3))
        
        if bdata is None:
            bdata = np.load('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Female1General_c3d/A1 - Stand_poses.npz')
        
        writer = self.writer
        gender_2_smpl_dict = self.gender_2_smpl_dict
        
        gender = str(bdata['gender'])
        
        betas = torch.Tensor(bdata['betas'])
        betas = betas.unsqueeze(0).to(self.device)
        
        smpl_dict = gender_2_smpl_dict[gender]
        
        shapedirs = smpl_dict['shapedirs']
        J_regressor = smpl_dict['J_regressor']
        v_template = smpl_dict['v_template']
        
        blended_shapes = blend_shapes(betas, shapedirs)
        v_shaped = v_template + blended_shapes
        J = vertices2joints(J_regressor, v_shaped)
        
        # full_pose = torch.Tensor(bdata['poses'])# [:,3:])
        # flatten_pose = full_pose.view(-1, 3)
        # rot_mats = batch_rodrigues(
        #     flatten_pose, dtype=torch.float32).view([full_pose.shape[0], -1, 3, 3])
        # rot_mats = torch.Tensor(rot_mats)
        rot_mats = torch.reshape(rot_mats, (times, -1, 3, 3)) # (T, N, 3, 3)

        rot_mat_np = (torch.flatten(rot_mats,0,1)).cpu().numpy()
        euler_angles_deg = R.from_matrix(rot_mat_np).as_euler('ZXY', degrees=True)
        euler_angles_deg = np.reshape(euler_angles_deg,(times,-1,3)) # (T, N, 3)
        
        offset = writer.cal_offset(J[0].cpu()) # (N, 3)
        
        num_joints = self.num_joints
        
        writer.write_bvh_from_offsets_rotations(offset[:num_joints], euler_angles_deg[:,:num_joints], root_motion, os.path.join(save_dir,filename))
        print('成功保存bvh文件')

    def get_pos_tn3(self,rot_mats,root_motion=None,bdata=None):
        # bdata: smplx .npz data
        # rot_mats: (t,52,9)
        # filename: path/to/yourfile.bvh
        # return (t,52,3) xyz
        times = rot_mats.shape[0]
        if bdata is None:
            bdata = np.load('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Female1General_c3d/A1 - Stand_poses.npz')
        
        gender_2_smpl_dict = self.gender_2_smpl_dict
        
        gender = str(bdata['gender'])
        
        betas = torch.Tensor(bdata['betas'])
        betas = betas.unsqueeze(0).to(self.device)
        
        smpl_dict = gender_2_smpl_dict[gender]
        
        shapedirs = smpl_dict['shapedirs']
        J_regressor = smpl_dict['J_regressor']
        v_template = smpl_dict['v_template']
        
        blended_shapes = blend_shapes(betas, shapedirs)
        v_shaped = v_template + blended_shapes
        J = vertices2joints(J_regressor, v_shaped)
        J = J.expand(times,-1,-1)
        
        parents = self.parents
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=torch.float32)
        
        if root_motion is None:
            # root_motion = bdata['trans']
            root_motion = torch.zeros((times,1,3)).to(J_transformed.device)
        J_transformed += root_motion
        
        return J_transformed

def down_sample(seq, fps, target_fps):
    # 采样率转换
    # seq: (T, D)
    # fps, target_fps: int
    # return: (T1, D)
    ratio = fps / target_fps
    indices = torch.arange(0, seq.shape[0], ratio).long()
    indices = torch.clamp(indices, max=seq.shape[0] - 1)
    return seq[indices]

def collate_fn_min_seq(batch):
    """
    batch: list of tuples, each is (rot_mats, text)
        rot_mats: (T, 52, 9)
        text: string
    returns:
        rot_mats_tensor: (batch_size, min_seq_len, 52, 9)
        text_list: list of str
    """
    # batch = [item for item in batch if item[0].shape[0] >= input_frame_len]
    # min_len = min(item[0].shape[0] for item in batch)
    # min_len = min(min_len,120) # window_len 为120
    rot_mats_batch = []
    text_batch = []
    for rot_mats, text in batch:
        start_time = 0
        rot_mats_batch.append(rot_mats)  # truncate to min_len
        text_batch.append(text)
    rot_mats_tensor = torch.stack(rot_mats_batch, dim=0)  # (B, min_len, 52, 9)
    return rot_mats_tensor, text_batch

def get_all_start_idx(seq_len,batch_len):
    if seq_len < batch_len:
        return None
    interval = batch_len // 4
    max_start_idx = (seq_len - batch_len)
    return np.arange(0,max_start_idx,interval)


# def 报告数据情况
def report_dataset(root_dir, datasetname):
    assert os.path.exists(root_dir)
    pose_times = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_poses.npz'):
                npz_path = os.path.join(subdir, file)
                bdata = np.load(npz_path)            
                full_pose = bdata['poses']
                fps = int(bdata['mocap_framerate'].item())
                pose_len = bdata['poses'].shape[0]
                pose_time = pose_len / fps
                pose_times.append(pose_time)
    # 绘制 pose_time 分布的柱状图（1 秒间隔）
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, max(pose_times) + 1, 1)
    plt.hist(pose_times, bins=bins, edgecolor='black')
    plt.xlabel('Pose Time (seconds)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Pose Time (1-second bins) dataset: {}'.format(datasetname))
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, datasetname+'.pdf')
    plt.savefig(save_path)
    plt.close()
    print('已保存数据集报告于 ' + save_path)



class AMASSDataset(Dataset):
    def __init__(self, root_dir, device, time_len=10, use_hand=False, target_fps=30, use_6d=False):
        """
        :param root_dir: 根目录，例如 'ACCAD'
        """
        self.time_len = time_len
        self.batch_len = time_len * target_fps
        self.root_dir = root_dir
        self.num_joints = 52 if use_hand else 21
        self.use_hand = use_hand
        self.target_fps = target_fps
        self.use_6d = use_6d
        self.samples = []
        if isinstance(root_dir, list):
            for root_dir_temp in root_dir:
                for subdir, _, files in os.walk(root_dir_temp):
                    for file in files:
                        if file.endswith('_poses.npz'):
                            npz_path = os.path.join(subdir, file)
                            bdata = np.load(npz_path)
                            
                            full_pose = bdata['poses']
                            fps = int(bdata['mocap_framerate'].item())
                            
                            full_pose = down_sample(full_pose, fps, target_fps)
                            pose_len = full_pose.shape[0]
                            
                            # 是下采样后的start_idx
                            start_idxs = get_all_start_idx(pose_len, self.time_len*target_fps)
                            if start_idxs is not None:
                                for start_idx in start_idxs:
                                    self.samples.append((npz_path,file,start_idx))
        else:
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('_poses.npz'):
                        npz_path = os.path.join(subdir, file)
                        bdata = np.load(npz_path)
                        
                        full_pose = bdata['poses']
                        fps = int(bdata['mocap_framerate'].item())
                        
                        full_pose = down_sample(full_pose, fps, target_fps)
                        pose_len = full_pose.shape[0]
                        
                        # 是下采样后的start_idx
                        start_idxs = get_all_start_idx(pose_len, self.time_len*target_fps)
                        if start_idxs is not None:
                            for start_idx in start_idxs:
                                self.samples.append((npz_path,file,start_idx))
        print('数据集创建成功，共 {} 条数据，单条数据时间长度 {} 秒，采样率 {}Hz。'.format(len(self.samples), self.time_len, target_fps))
        self.processor = Processor(device,use_hand=use_hand)
        
        if isinstance(root_dir, list):
            datasetname=[]
            for root_dir_temp in root_dir:
                datasetname.append(root_dir_temp.split('/')[-1])
            datasetname = '+'.join(datasetname)
            # report_dataset(root_dir='+'.join(root_dir), datasetname=datasetname)
        else:
            datasetname=root_dir.split('/')[-1]
            report_dataset(root_dir=root_dir, datasetname=datasetname)
        
        self.datasetname = datasetname
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        num_joints = self.num_joints
        npz_path, text, start_idx = self.samples[idx]
        bdata = np.load(npz_path)
        full_pose = torch.Tensor(bdata['poses']) # (T,N*3)
        
        fps = int(bdata['mocap_framerate'].item())
        target_fps = self.target_fps
        
        # 下采样
        full_pose = down_sample(full_pose, fps, target_fps)
        
        rot_mats = self.processor.fullpose_to_RotMat(full_pose) # (T,N,3,3)
        if self.use_6d:
            rot_mats = rot_mats[:,:,:2,:] # (T,N,2,3)
        rot_mats = torch.flatten(rot_mats,-2,-1)
        return rot_mats[start_idx:start_idx+self.batch_len, :num_joints], text  # (seq_len, num_joints, joint_size)

    def apply_train_valid_divide(self, train=True, split_ratio=0.9, seed=42):
        # mode: {0,1} 1为train, 0为valid
        # 在 root_dir 下寻找 divide.pkl文件
        # 有的话读取, 没有的话创建
        if isinstance(self.root_dir,list):
            divide_path = os.path.join('./train_valid_divide', '{}.pkl'.format(self.datasetname))
        else:
            divide_path = os.path.join(self.root_dir, 'divide.pkl')
        total = len(self.samples)
        all_indices = list(range(total))
        if os.path.exists(divide_path):
            with open(divide_path, 'rb') as f:
                divide_data = pickle.load(f)
            train_indices = divide_data['train']
            valid_indices = divide_data['valid']
            print('已读取 train valid 划分')
        else:
            random.seed(seed)
            random.shuffle(all_indices)
            split = int(split_ratio * total)
            train_indices = all_indices[:split]
            valid_indices = all_indices[split:]
            divide_data = {'train': train_indices, 'valid': valid_indices}
            with open(divide_path, 'wb') as f:
                pickle.dump(divide_data, f)
            print(f"创建train valid划分文件 {divide_path}")
        if train:
            self.samples = [self.samples[i] for i in train_indices]
        else:
            self.samples = [self.samples[i] for i in valid_indices]
        print(f"已加载 {'train' if train else 'valid'} 划分, 它有 {len(self.samples)} 个样本.")




# 帮我写一个函数, 获取循环姿态序列
# 对于一个骨骼序列
# pose: (...,N,3) 轴角制
# 设置最短序列长度T_m,在这之后的时间中寻找 测地线距离 最小的时刻T,截取到这个时刻作为循环时间
# 如果这个 测地线距离 > threshold, 返回None
# pose = pose[:T,N,3]
# offset = pose[0,...] - pose[-1,...]
# for k in range(1,T_m):
#    pose[-k,...] = pose[-k,...] + offset * k/T_m
# 返回pose

import torch.nn.functional as F

def axis_angle_to_matrix(pose):  # (T, N, 3) → (T, N, 3, 3)
    angle = torch.norm(pose, dim=-1, keepdim=True).clamp(min=1e-6)  # (T, N, 1)
    axis = pose / angle  # 单位轴
    x, y, z = axis.unbind(-1)  # 分解
    angle = angle.squeeze(-1)

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1 - cos

    # 构造罗德里格斯公式旋转矩阵
    R = torch.stack([
        cos + x * x * one_minus_cos,
        x * y * one_minus_cos - z * sin,
        x * z * one_minus_cos + y * sin,

        y * x * one_minus_cos + z * sin,
        cos + y * y * one_minus_cos,
        y * z * one_minus_cos - x * sin,

        z * x * one_minus_cos - y * sin,
        z * y * one_minus_cos + x * sin,
        cos + z * z * one_minus_cos,
    ], dim=-1).reshape(*pose.shape[:-1], 3, 3)
    return R


def geodesic_distance(R1, R2):  # (..., 3, 3)
    # R_diff = R1 * R2^T
    R_diff = torch.matmul(R1, R2.transpose(-1, -2))
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)  # (...,) 弧度


def get_looping_pose_geodesic(pose, T_m=10, threshold=0.3):  # threshold = max radians allowed
    """
    输入 pose: (T, N, 3)，返回循环对齐的姿态序列
    """
    T = pose.shape[0]
    if T <= T_m:
        return None

    R = axis_angle_to_matrix(pose)  # (T, N, 3, 3)
    R0 = R[0]  # (N, 3, 3)

    distances = []
    for t in range(T_m, T):
        Rt = R[t]  # (N, 3, 3)
        d = geodesic_distance(R0, Rt)  # (N,)
        distances.append(d.mean())  # 平均所有关节距离

    distances = torch.stack(distances)  # (T - T_m,)
    best_dist, best_idx = distances.min(dim=0)

    if best_dist > threshold:
        return None

    T_loop = T_m + best_idx.item() + 1
    new_pose = pose[:T_loop].clone()

    # 平滑末尾
    offset = new_pose[0] - new_pose[-1]
    for k in range(1, T_m + 1):
        new_pose[-k] += offset * (k / T_m)

    return new_pose



'''
device = torch.device('cuda')
train_set = AMASSDataset('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD', device)
bdata = np.load('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD/Female1Running_c3d/C4 - Run to walk1_poses.npz')
train_set.processor.write_bvh(rot_mats=None, bdata=bdata, filename='watch.bvh')
'''

'''
device = torch.device('cuda')
# watch -n 1 nvidia-smi

train_set = AMASSDataset('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD', device)
train_set.apply_train_valid_divide()

valid_set = AMASSDataset('/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/ACCAD', device)
valid_set.apply_train_valid_divide(train=False)


from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=8, collate_fn=collate_fn_min_seq, shuffle=True)
valid_loader = DataLoader(train_set, batch_size=8, collate_fn=collate_fn_min_seq, shuffle=False)
'''