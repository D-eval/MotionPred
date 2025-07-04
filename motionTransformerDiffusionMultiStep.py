# 多个噪声query，经过self-attention，再用crossAttention看到之前信息（的编码)


import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

import torch.fft

import sys

use_last_layernorm = True

# sys.path.append('/home/vipuser/DL/Dataset100G/DL-3D-Upload/sutra/tensorFlow-project/motion-transformer')
# from common.constants import Constants as C
# from common.conversions import compute_rotation_matrix_from_ortho6d



def normalize_vector(v, return_mag=False):
    # v: (batch_size, n)
    v_mag = torch.norm(v, dim=-1, keepdim=True)  # (batch_size, 1)
    v_mag = torch.clamp(v_mag, min=1e-6)
    v_normalized = v / v_mag
    if return_mag:
        return v_normalized, v_mag.squeeze(-1)
    else:
        return v_normalized


def cross_product(u, v):
    # u, v: (batch_size, 3)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.stack([i, j, k], dim=1)  # (batch_size, 3)
    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    # ortho6d: (B, 6) -> (B, 3, 3)
    x_raw = ortho6d[:, 0:3]  # (batch_size, 3)
    y_raw = ortho6d[:, 3:6]  # (batch_size, 3)

    x = normalize_vector(x_raw)           # (batch_size, 3)
    z = normalize_vector(cross_product(x, y_raw))  # (batch_size, 3)
    y = cross_product(z, x)               # (batch_size, 3)

    x = x.unsqueeze(-2)  # (batch_size,1, 3)
    y = y.unsqueeze(-2)
    z = z.unsqueeze(-2)
    matrix = torch.cat([x, y, z], dim=-2)  # (batch_size, 3, 3)
    return matrix


def scaled_dot_product_attention(q, k, v, mask, rel_key_emb=None, rel_val_emb=None, mask_type='look_ahead'):
    # attn_dim: num_joints for spatial and seq_len for temporal
    '''
    The scaled dot product attention mechanism introduced in the Transformer
    :param q: the query vectors matrix (b, h, T_q, D/h)
    :param k: the key vector matrix (b, h, T, D/h)
    :param v: the value vector matrix (b, h, T, D/h)
    :param mask: (T_q, T) or (B, T)
    :return: the updated encoding and the attention weights matrix
    '''
    # (b, h, T_q, D/h), (b, h, D/H, T) -> (b, h, T_q, T)
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))

    batch_size = q.shape[0]
    heads = q.shape[1]
    length = q.shape[2]
    
    if rel_key_emb is not None:
        q_t = q.permute(2,0,1,3)
        q_t_r = torch.reshape(q_t, (length, heads*batch_size, -1))
        q_tz_matmul = torch.matmul(q_t_r, rel_key_emb.transpose(-1,-2))
        q_tz_matmul_r = torch.reshape(q_tz_matmul, (length, batch_size, heads, -1))
        q_tz_matmul_r_t = q_tz_matmul_r.permute((1,2,0,3))
        matmul_qk += q_tz_matmul_r_t

    # scale matmul_qk
    dk = k.shape[-1] # int
    scaled_attention_logits = matmul_qk / math.sqrt(dk) # (B, H, T_q, T)

    # add the mask to the scaled tensor.
    if mask is not None:
        if mask_type=='look_ahead':
            # (B, H, T_q, T) + (T_q, T)
            scaled_attention_logits += (mask.to(dtype=scaled_attention_logits.dtype) * -1e9)
        elif mask_type=='random':
            # mask: (B, T) -> (B, 1, 1, T)，用于广播到 (B, H, T, T)
            attn_mask = mask.unsqueeze(1).unsqueeze(1).to(scaled_attention_logits.dtype)  # (B, 1, 1, T)
            scaled_attention_logits += attn_mask * -1e9
        else:
            raise ValueError('unknown mask type')
        
    # normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = F.softmax(scaled_attention_logits, -1) # (b,h,t+1,t+1)
    output = torch.matmul(attention_weights, v) # (b,h,t,t),(b,h,t,D/h) -> (b,h,t,D/h)
    
    if rel_val_emb is not None:
        w_t = attention_weights.permute((2,0,1,3))
        w_t_r = torch.reshape(w_t, (length, heads*batch_size, -1))
        w_tz_matmul = torch.matmul(w_t_r, rel_val_emb)
        w_tz_matmul_r = torch.reshape(w_tz_matmul, (length, batch_size, heads, -1))
        w_tz_matmul_r_t = w_tz_matmul_r.permute((1,2,0,3))
        output += w_tz_matmul_r_t

    return output, attention_weights



def get_angles(pos, i, d_model):
    # pos: (T,1) [[0],[1],...,[T-1]]
    # i: (1,D) [[0,1,...,D-1]]
    # d_model: int
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(window_len,d_model):
    # return: (1,T,1,D)
    angle_rads = get_angles(np.arange(window_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, :, np.newaxis, :]
    # return tf.cast(pos_encoding, dtype=tf.float32)  # (1, seq_len, 1, d_model)
    return pos_encoding # (window_len, d_model)


@staticmethod
def generate_relative_positions_matrix(length_q, length_k, max_relative_position):
        """
        Generates matrix of relative positions between inputs.
        Return a relative index matrix of shape [length_q, length_k]
        """
        # range_vec_k = tf.range(length_k)
        range_vec_k = torch.arange(10)
        range_vec_q = range_vec_k[-length_q:]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        # distance_mat_clipped = tf.clip_by_value(distance_mat,
        #                                         -max_relative_position,
        #                                         max_relative_position)
        distance_mat_clipped = torch.clip(distance_mat,-max_relative_position,max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a
        # relative position difference.
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat


def get_relative_embeddings(length_q, length_k):
    """
    Generates tensor of size [1 if cache else length_q, length_k, depth].
    """
    relative_positions_matrix = generate_relative_positions_matrix(length_q, length_k, max_relative_position)
    # key_emb = tf.gather(key_embedding_table, relative_positions_matrix)
    key_emb = key_embedding_table[relative_positions_matrix]
    # val_emb = tf.gather(value_embedding_table, relative_positions_matrix)
    val_emb = value_embedding_table[relative_positions_matrix]
    return key_emb, val_emb



def sep_split_heads(x, batch_size, seq_len, num_heads, d_model):
    '''
    split the embedding vector for different heads for the temporal attention
    :param x: the embedding vector (batch_size, seq_len, d_model)
    :param batch_size: batch size
    :param seq_len: sequence length
    :param num_heads: number of temporal heads
    :return: the split vector (batch_size, num_heads, seq_len, depth)
    '''
    depth = d_model // num_heads
    # x = tf.reshape(x, (batch_size, seq_len, num_heads, depth))
    x = torch.reshape(x,(batch_size, seq_len, num_heads, depth))
    # return tf.transpose(x, perm=[0, 2, 1, 3])
    return x.permute(0, 2, 1, 3)


def split_heads(x, b, shape1, attn_dim, num_heads, d_model):
    '''
    split the embedding vector for different heads for the spatial attention
    :param x: the embedding vector (batch_size, seq_len, num_joints, d_model)
    :param shape0: batch size
    :param shape1: sequence length
    :param attn_dim: number of joints
    :param num_heads: number of heads
    :return: the split vector (batch_size, seq_len, num_heads, num_joints, depth)
    '''
    depth = d_model // num_heads
    x = torch.reshape(x, (b, shape1, attn_dim, num_heads, depth))
    # return tf.transpose(x, perm=[0, 1, 3, 2, 4])
    return x.permute(0, 1, 3, 2, 4)


class Layernorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # scale
        self.beta = nn.Parameter(torch.zeros(d_model))  # shift
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D)
        # return: (B,T,D)
        mean = x.mean(dim=-1, keepdim=True)        # (B, T, 1)
        std = x.std(dim=-1, keepdim=True)          # (B, T, 1)
        normed = (x - mean) / (std + self.eps)      # 标准化
        out = self.gamma * normed + self.beta       # 仿射变换
        return out




class SlidingFourierEncoder(nn.Module):
    def __init__(self, window_len, stride=1, padding=0, mode='real_imag'):
        """
        使用滑动窗口的傅里叶变换器
        Args:
            window_len: 每个滑动窗口的长度
            stride: 滑动步长
            padding: 输入序列的 padding
            mode: 傅里叶特征模式 ['real_imag' | 'magnitude' | 'complex']
        """
        super().__init__()
        self.window_len = window_len
        self.stride = stride
        self.padding = padding
        self.freq_dim = window_len // 2 + 1
        assert mode in ['real_imag', 'magnitude', 'complex'], f"Unsupported mode: {mode}"
        self.mode = mode

    def forward(self, x):
        """
        Args:
            x: (..., 1, T)
        Returns:
            (..., D, T1), D depends on mode
        """
        orig_shape = x.shape[:-2]
        T = x.shape[-1]

        # padding
        x = F.pad(x, (self.padding, self.padding), mode='reflect')  # (..., 1, T_pad)

        # unfold: (..., 1, T_pad) -> (..., 1, T1, window_len)
        x_unfold = x.unfold(-1, self.window_len, self.stride)  # (..., 1, T1, window_len)
        x_unfold = x_unfold.squeeze(-4)  # (..., T1, window_len)

        # 进行 rfft
        x_fft = torch.fft.rfft(x_unfold, dim=-1)  # (..., T1, D)

        if self.mode == 'complex':
            x_out = x_fft.transpose(-2, -1)  # (..., D, T1), complex
        elif self.mode == 'real_imag':
            real = x_fft.real.transpose(-2, -1)  # (..., D, T1)
            imag = x_fft.imag.transpose(-2, -1)  # (..., D, T1)
            x_out = torch.cat([real, imag], dim=-2)  # (..., 2D, T1)
        elif self.mode == 'magnitude':
            mag = torch.abs(x_fft.transpose(-2, -1))  # (..., D, T1)
            x_out = mag
        return x_out

# 门控 top-C

class AttentionPool(nn.Module):
    def __init__(self,cls_num,d_model,joint_num):
        super().__init__()
        self.cls_num = cls_num
        self.joint_num = joint_num
        self.querys = nn.Parameter(torch.randn(joint_num, cls_num, d_model))  # (N, C, D)
        self.scale = d_model ** 0.5
    def forward(self,x):
        # x: (B,T,N,D)
        # return: (B,C,N,D)
        B, T, N, D = x.shape
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, cls_num, D)
        k = x  # (B, T, D)
        v = x
        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale  # (B, cls_num, T)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, cls_num, D)
        return out




class PosteriorRotationLayer(nn.Module):
    def __init__(self, d_model, num_heads, epsilon):
        # D=d_model, H=num_heads
        # let d = D / num_heads
        # enc_input: (B, t, d)
        # context: (B, d*(d-1)/2, D)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError('d_model必须被num_heads整除')
        depth = d_model // num_heads
        
        self.depth = depth
        self.linear_q = nn.Linear(depth,depth)
        
        self.tril_indices = torch.tril_indices(depth, depth, offset=-1)
        self.param_dim = (depth * (depth - 1)) // 2
        
        self.kv_context = nn.Linear(d_model, d_model)
        
        # layernorm
        self.layernorm = Layernorm(depth)
        
        # 旋转幅度
        self.epsilon = epsilon

    def forward(self, x, context=None):
        # x: (B, T, D)
        # context: (B, d*(d-1)/2, D)
        # return: (B, T, D)
        if context is None:
            return x  # No rotation
        
        B, T, D = x.shape
        H = self.num_heads
        d = self.depth
        if d*H != D:
            raise ValueError('x特征维度不符合,期望{},但是得到了{}'.format(d*H, D))
        
        x = self.split_head(x) # (B, H, T, d)
        q = self.linear_q(x) # (B, H, T, d)
        
        kv = self.kv_context(context) # (B, C, D)
        kv = self.split_head(kv) # (B, H, d*(d-1)/2, d)
        # 因为ctx不再处理了，kv几乎是其后来唯一的隐层
        kv = self.layernorm(kv)

        skew_params = torch.matmul(q, kv.transpose(-1,-2)) # (B, H, T, d*(d-1)/2)
        
        # 范围限制
        scale = 1.0  # 或者 0.5、0.1 等
        skew_params = scale * torch.tanh(skew_params / scale)
        
        # 构造反对称矩阵 (B, H, T, d, d)
        A = torch.zeros(B, H, T, d, d, device=q.device)
        A[:, :, :, self.tril_indices[0], self.tril_indices[1]] = skew_params
        A = A - A.transpose(-1, -2)  # 反对称

        epsilon = self.epsilon
        R = torch.eye(d, device=q.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) + epsilon * A  
        # (B, H, T, d, d)

        # (B, H, T, d)
        x = x.unsqueeze(-2) # (B, H, T, 1, d)
        # (B, H, T, 1, d) @ (B, H, T, d, d)
        x_rot = torch.matmul(x, R.transpose(-1, -2))  # (B, H, T, 1, d)
        x_rot = x_rot[...,0,:] # (B, H, T, d)
        x_rot = self.merge_head(x_rot)
        return x_rot

    def split_head(self,x):
        # x: (B,T,D)
        # return: (B,H,T,d)
        B, T, D = x.shape
        H = self.num_heads
        depth = self.depth
        y = torch.reshape(x, (B, T, H, depth))
        y = y.permute(0, 2, 1, 3)
        return y
    
    def merge_head(self, x):
        # x: (B, H, T, d) => (B, T, H * d) = (B, T, D)
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, H, d)
        x = x.view(B, T, H * d)           # (B, T, D)
        return x

# 消融实验中作为对比
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, x, context=None):
        # x: (B, T, D)
        # context: (B, C, D)
        # return: (B, T, D)
        # 作为对比，C = d*(d-1)/2
        if context is None:
            return x
        B, T, D = x.shape
        C = context.size(1)

        q = self.linear_q(x)      # (B, T, D)
        k = self.linear_k(context)  # (B, C, D)
        v = self.linear_v(context)  # (B, C, D)

        q = self.split_head(q)  # (B, H, T, d)
        k = self.split_head(k)  # (B, H, C, d)
        v = self.split_head(v)  # (B, H, C, d)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.depth ** 0.5)  # (B, H, T, C)
        attn = torch.softmax(attn_weights, dim=-1)  # (B, H, T, C)
        context_attended = torch.matmul(attn, v)    # (B, H, T, d)

        out = self.merge_head(context_attended)     # (B, T, D)
        out = self.linear_out(out)
        return out
    def split_head(self,x):
        # x: (B,T,D)
        # return: (B,H,T,d)
        B, T, D = x.shape
        H = self.num_heads
        depth = self.depth
        y = torch.reshape(x, (B, T, H, depth))
        y = y.permute(0, 2, 1, 3)
        return y
    def merge_head(self, x):
        # x: (B, H, T, d) => (B, T, H * d) = (B, T, D)
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, H, d)
        x = x.view(B, T, H * d)           # (B, T, D)
        return x


class SepTemporalAttention(nn.Module):
    def __init__(self,d_model, num_heads, shared_templ_kv,
                 temp_abs_pos_encoding, window_len,
                 num_joints, temp_rel_pos_encoding):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.shared_templ_kv=shared_templ_kv
        self.temp_abs_pos_encoding=temp_abs_pos_encoding
        self.window_len=window_len
        self.num_joints=num_joints
        pos_encoding = positional_encoding(self.window_len,self.d_model)
        pos_encoding = torch.Tensor(pos_encoding)
        self.register_buffer('pos_encoding',pos_encoding)
        self.temp_rel_pos_encoding = temp_rel_pos_encoding
        if self.shared_templ_kv:
            self.linear_k_all = nn.Linear(self.d_model,self.d_model)
            self.linear_v_all = nn.Linear(self.d_model,self.d_model)
        else:
            self.linear_k = nn.ModuleList([
                nn.Linear(self.d_model,self.d_model)
                for _ in range(self.num_joints)
            ])
            self.linear_v = nn.ModuleList([
                nn.Linear(self.d_model,self.d_model)
                for _ in range(self.num_joints)
            ])
        self.linear_q = nn.ModuleList([
            nn.Linear(self.d_model,self.d_model)
            for _ in range(self.num_joints)
        ])
        self.linear_output = nn.Linear(self.d_model,self.d_model)
    def forward(self,x,mask,mask_type):
        # x: (B, T, N, D)
        # mask: (T, T) or (B, T) 自己咋算都行，别让别人看到你就行了
        # return: (B, T, N, D)
        if self.temp_abs_pos_encoding:
            inp_seq_len = x.shape[1]
            x[:,:] += self.pos_encoding[:, :inp_seq_len]
        outputs = []
        attn_weights = []
        batch_size,seq_len,num_joints,d_model = x.shape
        x = x.permute(2,0,1,3) # (N, B, T, D)
        if self.shared_templ_kv:
            k_all = self.linear_k_all(x) # (n,b,t,D)
            v_all = self.linear_v_all(x) # (n,b,t,D)
        rel_key_emb, rel_val_emb = None, None
        if self.temp_rel_pos_encoding: # False
            rel_key_emb, rel_val_emb = get_relative_embeddings(seq_len, seq_len)
        # different joints have different embedding matrices
        for joint_idx in range(self.num_joints):
            joint_rep = x[joint_idx]  # (b,t+1,d)
            q = self.linear_q[joint_idx](joint_rep) # (b,t+1,D)
            if self.shared_templ_kv:
                v = v_all[joint_idx]
                k = k_all[joint_idx]
            else:
                k = self.linear_k[joint_idx](joint_rep)
                v = self.linear_v[joint_idx](joint_rep)
                # k_ctx = self.linear_k_context[joint_idx](context)
                # v_ctx = self.linear_v_context[joint_idx](context)
                # (b,D)
            # (b,D)
            # split it to several attention heads
            q = sep_split_heads(q, batch_size, seq_len, self.num_heads, self.d_model)
            # (b, h, t, D/h)
            k = sep_split_heads(k, batch_size, seq_len, self.num_heads, self.d_model)
            # (b, h, t, D/h)
            v = sep_split_heads(v, batch_size, seq_len, self.num_heads, self.d_model)
            # (b, h, t, D/h)
            # calculate the updated encoding by scaled dot product attention
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, rel_key_emb, rel_val_emb, mask_type)
            # (b,h,t,D/h)
            scaled_attention = scaled_attention.permute(0, 2, 1, 3)
            # (b,t,h,D/h)
            # flatten h,D/h
            concat_attention = torch.reshape(scaled_attention, (batch_size, seq_len, d_model))
            # (b,t,D)
            output = self.linear_output(concat_attention)
            outputs += [output.unsqueeze(2)]
            # (b, t, 1, D)
            # 只取最后一个注意力层的weight
            last_attention_weights = attention_weights[:, :, -1, :]  # (batch_size, num_heads, seq_len)
            attn_weights += [last_attention_weights]
        # 拼接 num_joints
        outputs = torch.cat(outputs,dim=2)
        # (b,t,n,D)
        attn_weights = torch.stack(attn_weights,dim=1)  # (batch_size, num_joints, num_heads, seq_len)
        return outputs, attn_weights


class SepSpacialAttention(nn.Module):
    def __init__(self,d_model,num_joints,num_heads_spacial,
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints
        self.num_heads_spacial = num_heads_spacial
        self.linear_key = nn.Linear(self.d_model,self.d_model)
        self.linear_value = nn.Linear(self.d_model,self.d_model)
        self.linear_query = nn.ModuleList([
            nn.Linear(self.d_model,self.d_model)
            for _ in range(self.num_joints)
        ])
        self.linear_output = nn.Linear(self.d_model,self.d_model)
    def forward(self,x, mask=None):
        # x: (batch_size, seq_len, num_joints, d_model)
        # mask: None
        k = self.linear_key(x)
        v = self.linear_value(x)
        x = x.permute(2,0,1,3) # (N,B,L,D)
        q_joints = []
        for joint_idx in range(self.num_joints):
            q = self.linear_query[joint_idx](x[joint_idx]) # (B,L,D)
            q = q.unsqueeze(2) # (B,L,1,D)
            q_joints += [q]
        q_joints = torch.cat(q_joints,dim=2) # (B,L,N,D)
        batch_size,seq_len,_,_ = q_joints.shape
        # (batch_size,seq_len,num_heads,num_joints,depth) where depth = d_model / num_heads
        q_joints = split_heads(q_joints, batch_size, seq_len,
                            self.num_joints, self.num_heads_spacial,self.d_model)
        k = split_heads(k, batch_size, seq_len,
                        self.num_joints, self.num_heads_spacial,self.d_model)
        v = split_heads(v, batch_size, seq_len,
                        self.num_joints, self.num_heads_spacial,self.d_model)
        scaled_attention, attention_weights = scaled_dot_product_attention(q_joints, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 1, 3, 2, 4)
        concat_attention = torch.reshape(scaled_attention, (batch_size, seq_len, self.num_joints, self.d_model))
        output = self.linear_output(concat_attention)
        attention_weights = attention_weights[:, -1, :, :, :]  # (batch_size, num_heads, num_joints, num_joints)
        return output, attention_weights



class SepTemporalCrossAttention(nn.Module):
    def __init__(self,d_input, d_model, num_heads, shared_templ_kv,
                 temp_abs_pos_encoding, window_len,
                 num_joints, temp_rel_pos_encoding):
        super().__init__()
        self.d_input=d_input
        self.d_model=d_model
        self.num_heads=num_heads
        self.shared_templ_kv=shared_templ_kv
        self.temp_abs_pos_encoding=temp_abs_pos_encoding
        self.window_len=window_len
        self.num_joints=num_joints
        # pos_encoding = positional_encoding(self.window_len,self.d_model) # (1, T, 1, D)
        # self.pos_encoding = torch.Tensor(pos_encoding)
        self.temp_rel_pos_encoding = temp_rel_pos_encoding
        if self.shared_templ_kv:
            self.linear_k_all = nn.Linear(self.d_model,self.d_model)
            self.linear_v_all = nn.Linear(self.d_model,self.d_model)
        else:
            self.linear_k = nn.ModuleList([
                nn.Linear(self.d_input,self.d_model)
                for _ in range(self.num_joints)
            ])
            self.linear_v = nn.ModuleList([
                nn.Linear(self.d_input,self.d_model)
                for _ in range(self.num_joints)
            ])
        self.linear_q = nn.ModuleList([
            nn.Linear(self.d_model,self.d_model)
            for _ in range(self.num_joints)
        ])
        self.linear_output = nn.Linear(self.d_model,self.d_model)
        time_encoding = torch.Tensor(positional_encoding(self.window_len,self.d_model)) # (1,500,1,D)
        self.register_buffer('time_encoding',time_encoding)
    def forward(self,eps_v,history_pos,mask):
        # eps_v: (B, T_q, N, D), noised v
        # history_pos: (B, T, N, D)
        # mask: (T_q, T) look ahead mask
        # return: (B, T_q, N, D)
        B,T,N,d = history_pos.shape
        _,T_q,_,D = eps_v.shape # 只能为0和T
        if T_q==T:
            if self.temp_abs_pos_encoding:
                eps_v[:,:] += self.time_encoding[:, :T]
        elif T_q==1:
            if self.temp_abs_pos_encoding:
                eps_v[:,0] += self.time_encoding[:, T]
        else:
            if self.temp_abs_pos_encoding:
                eps_v[:,:] += self.time_encoding[:, :T_q]
            # raise ValueError("eps_v 长度是")
        outputs = []
        attn_weights = []
        eps_v = eps_v.permute(2,0,1,3) # (N, B, T_q, D)
        history_pos = history_pos.permute(2,0,1,3) # (N, B, T_q, D)
        if self.shared_templ_kv:
            k_all = self.linear_k_all(history_pos) # (N,B,T,D)
            v_all = self.linear_v_all(history_pos) # (N,B,T,D)
        rel_key_emb, rel_val_emb = None, None
        if self.temp_rel_pos_encoding: # False
            rel_key_emb, rel_val_emb = get_relative_embeddings(T, T)
        # different joints have different embedding matrices
        for joint_idx in range(self.num_joints):
            joint_rep = eps_v[joint_idx]  # (B,T_q,D)
            joint_rep_history = history_pos[joint_idx]  # (B,T,D)
            # 这里还要加时间编码
            # joint_rep_history += self.time_encoding[:,:T,0,:]
            q = self.linear_q[joint_idx](joint_rep) # (B,T_q,D)
            if self.shared_templ_kv:
                v = v_all[joint_idx]
                k = k_all[joint_idx]
            else:
                k = self.linear_k[joint_idx](joint_rep_history) # (B,T,D)
                v = self.linear_v[joint_idx](joint_rep_history)
            # k += self.time_encoding[:,:T,0,:]
            # v += self.time_encoding[:,:T,0,:]
            # split it to several attention heads
            q = sep_split_heads(q, B, T_q, self.num_heads, self.d_model)
            # (B, H, T_q, D/h)
            k = sep_split_heads(k, B, T, self.num_heads, self.d_model)
            v = sep_split_heads(v, B, T, self.num_heads, self.d_model)
            # (B, H, T, D/h)
            # calculate the updated encoding by scaled dot product attention
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, rel_key_emb, rel_val_emb, 'look_ahead')
            # (B, H, T_q, D/h)
            scaled_attention = scaled_attention.permute(0, 2, 1, 3)
            # (B, T_q, H, D/h)
            concat_attention = torch.reshape(scaled_attention, (B, T_q, D))
            # (B, T_q, D)
            output = self.linear_output(concat_attention) # 这里不用 joint_wise 的linear
            outputs += [output]
            # (B, T, D)
            # 只取最后一个注意力层的weight
            last_attention_weights = attention_weights[:, :, -1, :]  # (batch_size, num_heads, seq_len)
            attn_weights += [last_attention_weights]
        # 拼接 num_joints
        outputs = torch.stack(outputs, dim=2) # stack的dim是插空
        # (B, T_q, N, D)
        attn_weights = torch.stack(attn_weights,dim=1)  # (B,N,H,T_q)
        return outputs, attn_weights



class PointWiseFeedForward(nn.Module):
    def __init__(self,d_model,num_joints):
        super().__init__()
        self.d_model=d_model
        self.num_joints=num_joints
        self.ff1 = nn.ModuleList([
            nn.Linear(self.d_model,self.d_model)
            for _ in range(self.num_joints)
        ])
        self.ff2 = nn.ModuleList([
            nn.Linear(self.d_model,self.d_model)
            for _ in range(self.num_joints)
        ])
    def forward(self,inputs):
        # inputs: (batch_size, seq_len, num_joints, d_model)
        inputs = inputs.permute((2, 0, 1, 3))
        outputs = []
        for idx in range(self.num_joints):
            joint_outputs = F.relu(self.ff1[idx](inputs[idx]))
            joint_outputs = self.ff2[idx](joint_outputs)
            outputs += [joint_outputs]
        outputs = torch.cat(outputs, dim=-1)
        outputs = torch.reshape(outputs, (outputs.shape[0], outputs.shape[1], self.num_joints, self.d_model))
        return outputs


class CrossAttentionN(nn.Module):
    def __init__(self,num_joints,num_heads,d_model):
        super().__init__()
        num_joints = num_joints
        num_heads = num_heads
        d_model = d_model
        
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model
        self.num_joints = num_joints
        assert d_model % num_heads == 0
        
        self.linear_q = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_joints)
        ])
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
    def forward(self,x,context=None):
        # x: (B, T, N, D)
        # context: (B, C, D)
        # return: (B, T, N, D)
        if context is None:
            return x
        B, T, N, D = x.shape
        x = x.permute(2,0,1,3) # (N, B, T, D)
        assert N == self.num_joints
        
        C = context.size(1)

        qs = [
            self.linear_q[joint_idx](x[joint_idx]) # (B,T,D)
            for joint_idx in range(N)
        ]
        outs = []
        for joint_idx in range(N):
            q = qs[joint_idx] # (B, T, D)
            k = self.linear_k(context)  # (B, C, D)
            v = self.linear_v(context)  # (B, C, D)

            q = self.split_head(q)  # (B, H, T, d)
            k = self.split_head(k)  # (B, H, C, d)
            v = self.split_head(v)  # (B, H, C, d)

            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.depth ** 0.5)  # (B, H, T, C)
            attn = torch.softmax(attn_weights, dim=-1)  # (B, H, T, C)
            context_attended = torch.matmul(attn, v)    # (B, H, T, d)

            out = self.merge_head(context_attended)     # (B, T, D)
            out = self.linear_out(out)
            outs += [out.unsqueeze(2)]
        outs = torch.cat(outs,dim=2) # (B,T,N,D)
        return outs
    def split_head(self,x):
        # x: (B,T,D)
        # return: (B,H,T,d)
        B, T, D = x.shape
        H = self.num_heads
        depth = self.depth
        y = torch.reshape(x, (B, T, H, depth))
        y = y.permute(0, 2, 1, 3)
        return y
    def merge_head(self, x):
        # x: (B, H, T, d) => (B, T, H * d) = (B, T, D)
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, H, d)
        x = x.view(B, T, H * d)           # (B, T, D)
        return x


class PosteriorRotationN(nn.Module):
    def __init__(self,num_joints,D_decoder,D_encoder,
                 epsilon, tanh_scale):
        super().__init__()
        num_joints = num_joints
        epsilon = epsilon
        tanh_scale = tanh_scale
        
        self.num_joints = num_joints
        self.epsilon = epsilon
        self.tanh_scale = tanh_scale
        
        self.linear_qs = nn.ModuleList([
            nn.Linear(D_encoder, D_decoder)
            for _ in range(num_joints)
        ])
        
        self.tril_indices = torch.tril_indices(D_decoder, D_decoder, offset=-1)
        self.param_dim = (D_decoder * (D_decoder - 1)) // 2
        
        self.kv_context = nn.Linear(D_encoder, D_decoder) # (B, d*(d-1)/2, D)
        
        # layernorm
        # self.layernorm = nn.LayerNorm(depth) # Layernorm(depth)
    def forward(self,x,context=None):
        # x: (B, T, N, D2) 隐层
        # context: (B, C, N, D1) where C = D2*(D2-1)/2
        # return: (B, T, N, D2)
        B,T,N,D2 = x.shape
        C = context.shape[1]
        # 先linear，再split_head
        kv = self.kv_context(context) # (B, C, N, D2)
        
        kv = kv.permute(0,2,3,1) # (B,N,D,C)
        x = x.permute(0,2,1,3) # (B,N,T,D)
        
        # (B,N,T,D) @ (B,N,D,C) -> (B,N,T,C)
        skew_params = x @ kv # (B,N,T,C)
        skew_params = self.epsilon * self.tanh_scale * torch.tanh(skew_params / self.tanh_scale)

        A = torch.zeros((B,N,T,D2,D2), device=x.device)
        A[..., self.tril_indices[0], self.tril_indices[1]] = skew_params
        A = A - A.transpose(-1,-2)
        
        dx = x[:,:,:,None,:] @ A # (B,N,T,1,D)

        y = x + dx
        y = y[:,:,:,0,:] # (B,N,T,D)
        y = y.permute(0,2,1,3) # (B,T,N,D2)
        
        return y


def get_model_cls(model_name):
    if model_name == 'CrossAttention':
        return CrossAttentionN
    elif model_name == 'PosteriorRotation':
        return PosteriorRotationN
    else:
        raise ValueError('未知的模型名称')


class ParaTransformerLayer(nn.Module):
    def __init__(self,d_model, num_head_spacial, num_head_temporal, 
                 num_joints, dropout_rate,
                 shared_templ_kv, temp_abs_pos_encoding,
                 window_len, temp_rel_pos_encoding,
                 use_posteriors, posterior_name,
                 num_heads_posterior,
                 epsilon, tanh_scale):
        super().__init__()
        self.d_model = d_model
        self.num_head_spacial = num_head_spacial
        self.num_head_temporal = num_head_temporal
        self.num_joints = num_joints
        self.dropout_rate = dropout_rate
        self.temporal_attn = SepTemporalAttention(d_model,num_head_temporal,shared_templ_kv,
                                                  temp_abs_pos_encoding, window_len,
                                                  num_joints, temp_rel_pos_encoding)
        self.spatial_attn = SepSpacialAttention(d_model,num_joints,num_head_spacial)
        
        self.feed_forward = PointWiseFeedForward(d_model,num_joints)

        self.dropout_temporal = nn.Dropout(self.dropout_rate)
        self.dropout_spatial = nn.Dropout(self.dropout_rate)
        
        self.dropout_ff = nn.Dropout(self.dropout_rate)

        self.ln_temporal = nn.LayerNorm(self.d_model)
        self.ln_spatial = nn.LayerNorm(self.d_model)
        self.ln_ff = nn.LayerNorm(self.d_model)

        if use_posteriors:
            if posterior_name=='CrossAttentionN':
                self.posterior_correction = CrossAttentionN(num_joints, num_heads_posterior, d_model)
            elif posterior_name=='PosteriorRotationN':
                self.posterior_correction = PosteriorRotationN(num_joints, num_heads_posterior, d_model,
                                                               epsilon, tanh_scale)
            else:
                raise ValueError('unknown model')
            self.dropout_posterior = nn.Dropout(self.dropout_rate)
            self.ln_posterior = nn.LayerNorm(self.d_model)
        self.use_posteriors = use_posteriors

    def forward(self, x, mask, context, mask_type):
        # x : (B, T, N, D)
        # context: (b, C, D)
        # mask : (T, T) or (B, T)
        # mask_type: 'look_ahead' or 'random'
        # return: (B, T, N, D)
        # temporal
        # x: (B, T, N, D)
        attn1, attn_weights_block1 = self.temporal_attn(x, mask, mask_type)
        # attn1: (B, T, N, D)
        attn1 = self.dropout_temporal(attn1)
        temporal_out = self.ln_temporal(attn1 + x)
        # spatial
        attn2, attn_weights_block2 = self.spatial_attn(x)
        attn2 = self.dropout_spatial(attn2)
        spatial_out = self.ln_spatial(attn2 + x)
        
        out = temporal_out + spatial_out
        
        # 后验修正
        if self.use_posteriors:
            out_posterior = self.posterior_correction(out, context=context)
            out_posterior = self.dropout_posterior(out_posterior)
            out = self.ln_posterior(out_posterior + out)

        # feed forward
        ffn_output = self.feed_forward(out)
        ffn_output = self.dropout_ff(ffn_output)
        final = self.ln_ff(ffn_output + out)

        return final, attn_weights_block1, attn_weights_block2




class ParaTransformerDecoderLayer(nn.Module):
    def __init__(self, d_input, d_model, num_head_spacial, num_head_temporal, 
                 num_joints, dropout_rate,
                 shared_templ_kv, temp_abs_pos_encoding,
                 window_len, temp_rel_pos_encoding,
                 use_posteriors, posterior_name,
                 num_heads_posterior,
                 epsilon, tanh_scale):
        super().__init__()
        self.d_model = d_model
        self.num_head_spacial = num_head_spacial
        self.num_head_temporal = num_head_temporal
        self.num_joints = num_joints
        self.dropout_rate = dropout_rate
        self.temporal_cross_attn = SepTemporalCrossAttention(d_input,d_model,num_head_temporal,shared_templ_kv,
                                                  temp_abs_pos_encoding, window_len,
                                                  num_joints, temp_rel_pos_encoding)
        self.spatial_attn = SepSpacialAttention(d_model,num_joints,num_head_spacial)
        
        self.feed_forward = PointWiseFeedForward(d_model,num_joints)

        self.dropout_temporal = nn.Dropout(self.dropout_rate)
        self.dropout_spatial = nn.Dropout(self.dropout_rate)
        self.dropout_ff = nn.Dropout(self.dropout_rate)

        self.ln_temporal = nn.LayerNorm(self.d_model)
        self.ln_spatial = nn.LayerNorm(self.d_model)
        self.ln_ff = nn.LayerNorm(self.d_model)

        # self attn
        self.temporal_self_attn = SepTemporalAttention(d_model,num_head_temporal,shared_templ_kv,
                                                  temp_abs_pos_encoding, window_len,
                                                  num_joints, temp_rel_pos_encoding)
        self.dropout_merge = nn.Dropout(self.dropout_rate)
        self.ln_merge = nn.LayerNorm(self.d_model)
        
        if use_posteriors:
            if posterior_name=='CrossAttentionN':
                self.posterior_correction = CrossAttentionN(num_joints, num_heads_posterior, d_model)
            elif posterior_name=='PosteriorRotationN':
                self.posterior_correction = PosteriorRotationN(num_joints, d_model, d_model,
                                                               epsilon, tanh_scale)
            else:
                raise ValueError('unknown model')
            # self.dropout_posterior = nn.Dropout(self.dropout_rate)
            # self.ln_posterior = nn.LayerNorm(self.d_model)
        self.use_posteriors = use_posteriors

    def forward(self, eps_v, history_pose, mask, context, use_self_layernorm=True):
        # eps_v: (B, T_q, N, D) # T_q间进行self_attn
        # history_pose: (B, T, N, D)
        # context: (B, C, N, D)
        # mask : (T_q, T)
        # return: (B, T_q, N, D)
        T_q = eps_v.shape[1]
        # cross attn
        attn1, attn_weights_block1 = self.temporal_cross_attn(eps_v, history_pose, mask)
        # attn1: (B, T_q, N, D)
        attn1 = self.dropout_temporal(attn1)
        temporal_out = self.ln_temporal(attn1 + eps_v)
        
        # spatial
        attn2, attn_weights_block2 = self.spatial_attn(eps_v)
        attn2 = self.dropout_spatial(attn2)
        if use_self_layernorm:
            spatial_out = self.ln_spatial(eps_v + attn2)
        else:
            spatial_out = eps_v + attn2
        
        # out = spatial_out
        
        # self attn
        # 既然要在像素上连续，就不要有attention，直接用卷积就行
        # mask_self_attn = torch.zeros((T_q,T_q)).to(eps_v.device)
        # attn3, attn_weights_block3 = self.temporal_self_attn(eps_v, mask_self_attn, "look_ahead") # 并非look_ahead，只是这样写而已。
        # attn3 = self.dropout_merge(attn3)
        # if use_self_layernorm:
        #     self_out = self.ln_merge(attn3 + eps_v)
        # else:
        #     self_out = attn3 + eps_v
        # 
        out = temporal_out # + spatial_out # + self_out

        # 后验修正
        if self.use_posteriors:
            out_posterior = self.posterior_correction(out, context=context)
            # out_posterior = self.dropout_posterior(out_posterior)
            # out = self.ln_posterior(out_posterior + out)

        # feed forward
        ffn_output = self.feed_forward(out)
        ffn_output = self.dropout_ff(ffn_output)
        final = self.ln_ff(ffn_output + out)

        return final, attn_weights_block1, attn_weights_block2



def create_look_ahead_mask(window_len):
    # shape: (window_len, window_len)
    # return: 下三角,不包含对角线 (大家都能看到自己)
    return torch.triu(torch.ones(window_len, window_len), diagonal=1)


# Decoder, 之后最好加上 diffusion
class TransformerDecoder(nn.Module):
    def __init__(self,num_joints, d_model, window_len,
                 abs_pos_encoding, num_layers, dropout_rate,
                 use_6d_outputs, joint_size, residual_velocity,
                 loss_type,
                 num_head_spacial,num_head_temporal,
                 shared_templ_kv,
                 temp_abs_pos_encoding,temp_rel_pos_encoding,
                 use_posteriors,
                 posterior_name,num_heads_posterior,
                 epsilon,tanh_scale,
                 need_output_bias):
        super().__init__()
        
        self.num_joints = num_joints
        self.d_model = d_model
        
        self.window_len = window_len
        look_ahead_mask = create_look_ahead_mask(self.window_len)
        self.register_buffer('look_ahead_mask',look_ahead_mask)
        self.abs_pos_encoding = abs_pos_encoding
        self.num_layers = num_layers
        
        self.dropout_rate = dropout_rate
        self.use_6d_outputs = use_6d_outputs
        self.joint_size = joint_size
        
        self.residual_velocity = residual_velocity
        self.loss_type = loss_type
        
        self.embeddings = nn.ModuleList([
            nn.Linear(self.joint_size,self.d_model)
            for _ in range(self.num_joints)
        ])
        pos_encoding = positional_encoding(self.window_len,self.d_model)
        self.register_buffer('pos_encoding', torch.Tensor(pos_encoding))
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.para_transformer_layers = nn.ModuleList([
            ParaTransformerDecoderLayer(d_model, d_model, num_head_spacial, num_head_temporal, 
                 num_joints, dropout_rate,
                 shared_templ_kv, temp_abs_pos_encoding,
                 window_len, temp_rel_pos_encoding,
                 use_posteriors, posterior_name,
                 num_heads_posterior,
                 epsilon, tanh_scale)
            for _ in range(self.num_layers)
        ])
        if not self.use_6d_outputs:
            self.output_linears = nn.ModuleList([
                nn.Linear(self.d_model,self.joint_size)
                for _ in range(self.num_joints)
            ])
        else:
            self.output_linears = nn.ModuleList([
                nn.Linear(self.d_model,6, bias=need_output_bias)
                for _ in range(self.num_joints)
            ])
        self.tau_gate_proj = nn.Linear(self.d_model, d_model) #
        
        # self.history_embedding = nn.ModuleList([
        #     nn.Linear(self.d_model,self.d_model)
        #     for _ in range(self.num_joints)
        # ])
            
    def forward(self, poses_tau, prior_context, tau_emb, context=None, mask=None):
        # poses_tau: (B, T_q, N, d_in)
        # prior_context: (B, T, N, D) 历史姿态的编码
        # tau_emb: (B, D)
        # context: (B, C, D)
        # return: noise_pred: (B, T_q, N, d_in)
        B, T, N, D = prior_context.shape
        _, T_q, _, d = poses_tau.shape
        poses_tau = poses_tau.permute(2,0,1,3) # (N,B,T_q,d)
        # prior_context = prior_context.permute(2,0,1,3) # (N,B,T,D)
        embed = []
        for joint_idx in range(self.num_joints):
            joint_rep = self.embeddings[joint_idx](poses_tau[joint_idx]) # (B,T,D)
            embed += [joint_rep]
            
        eps_pose = torch.stack(embed) # (N,B,T_q,D)
        eps_pose = eps_pose.permute(1,2,0,3) # (B,T_q,N,D)
        
        # history_embed = []
        # for joint_idx in range(self.num_joints):
        #     joint_rep = self.history_embedding[joint_idx](prior_context[joint_idx]) # (B,T,D)
        #     history_embed += [joint_rep]
        # 
        # history_embed = torch.stack(history_embed) # (N,B,T,D)
        # history_embed = history_embed.permute(1,2,0,3) # (B,T,N,D)
        

        if self.abs_pos_encoding: # 必须填True
            eps_pose += self.pos_encoding[:, T:T+T_q]
            # history_embed += self.pos_encoding[:,:T] # 已经在 history_encoder 中加入过了
            
        gate = torch.sigmoid(self.tau_gate_proj(tau_emb)) # (B, D)
        eps_pose = eps_pose + gate[:, None, None, :] * tau_emb[:, None, None, :]
        
        eps_pose = self.input_dropout(eps_pose) # (B,T_q,N,D)

        # put into several attention layers
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}
        
        # look_ahead_mask 是下三角，对角线为0
        look_ahead_mask = self.look_ahead_mask[:T, :T] if mask is None else mask
        for i in range(self.num_layers):
            # use_self_layernorm
            if (i < self.num_layers-1):
                eps_pose, block1, block2 = self.para_transformer_layers[i](eps_pose, prior_context, look_ahead_mask, context)
            else:
                eps_pose, block1, block2 = self.para_transformer_layers[i](eps_pose, prior_context, look_ahead_mask, context, use_self_layernorm=False)
            attention_weights_temporal += [block1]  # (B, N, H, T)
            attention_weights_spatial += [block2]  # (B, H, N, N)
        # (B,T,N,D)
        
        attention_weights['temporal'] = torch.stack(attention_weights_temporal, axis=1)  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        attention_weights['spatial'] = torch.stack(attention_weights_spatial, axis=1)  # (batch_size, num_layers, num_heads, num_joints, num_joints)

        # decode each feature to the rotation matrix space
        # different joints have different decoding matrices
        if not self.use_6d_outputs:
            # (N,B,T_q,D)
            eps_pose = eps_pose.permute(2,0,1,3)
            output = []
            for joint_idx in range(self.num_joints):
                joint_output = self.output_linears[joint_idx](eps_pose[joint_idx])
                output += [joint_output]

            final_output = torch.cat(output, dim=-1)
            final_output = torch.reshape(final_output, [final_output.shape[0],
                                                        final_output.shape[1],
                                                        self.num_joints,
                                                        self.joint_size])
        else:
            # (N, B, T, d)
            eps_pose = eps_pose.permute(2,0,1,3)
            output = []
            for joint_idx in range(self.num_joints):
                joint_output = self.output_linears[joint_idx](eps_pose[joint_idx]) # (B, T, d)
                output += [joint_output]
            final_output = torch.stack(output, dim=2)
        if self.residual_velocity: # 这里指的是扩散时间tau velocity，不是现实时间t velocity
            final_output += poses_tau.permute(1,2,0,3)
        return final_output, attention_weights

    def get_loss(self, data, return_each=False):
        # data: (B,seq_len,num_joints,joint_size)
        b,seq_len,num_joints,joint_size = data.shape
        seq_len = seq_len-1
        # look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len))
        # look_ahead_mask = look_ahead_mask.to(data.device)
        # look_ahead_mask = self.look_ahead_mask[:seq_len,:seq_len]
        inputs = data[:,:-1,...]
        targets = data[:,1:,...]
        output, _ = self.forward(inputs)
        diff = targets - output # (B,seq_len,num_joints,joint_size)
        if self.loss_type == 'all_mean':
            loss = (diff ** 2).mean(dim=(1, 2, 3))
            # (B,)
        elif self.loss_type == 'joint_sum':
            per_joint_loss = diff.view(-1, seq_len, self.num_joints, self.joint_size)
            per_joint_loss = torch.sqrt(torch.sum(per_joint_loss ** 2, dim=-1))  # (batch, seq_len, num_joints)
            per_joint_loss = torch.sum(per_joint_loss, dim=-1)  # sum over joints -> (batch, seq_len)
            loss = torch.mean(per_joint_loss,dim=-1)  # average over batch and seq_len
            # (B,)
        elif self.loss_type == 'geodesic': # 测地线
            # tmd, 太坑了
            if self.joint_size != 9:
                raise ValueError('geodesic loss expect joint_size 9, got [{}],\nplease use "all_mean" or "joint_sum"'.format(self.joint_size))
            target_angles = targets.view(-1, seq_len, self.num_joints, 3, 3)
            predicted_angles = output.view(-1, seq_len, self.num_joints, 3, 3)
            # 转置 predicted: (batch, seq_len, joints, 3, 3)
            pred_transpose = predicted_angles.transpose(-1, -2)
            # 矩阵乘法: R1 * R2^T
            m = torch.matmul(target_angles, pred_transpose)  # (..., 3, 3)
            # 计算 trace
            trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
            cos = (trace - 1) / 2
            # clamp 防止数值误差引起 nan
            eps = 1e-6
            cos = torch.clamp(cos, -1.0+eps, 1.0-eps)
            theta = torch.acos(cos)  # (batch, seq_len, joints)
            per_joint_loss = torch.sum(theta, dim=-1)  # sum over joints
            # (B,seq_len)
            per_joint_loss = torch.sum(per_joint_loss, dim=-1)  # sum over seq
            # (B,)
            loss = per_joint_loss # = torch.mean(per_joint_loss)  # average over batch
            # (B,)
        else:
            raise ValueError("unknown loss_type")
        assert not torch.isnan(loss).any(), "loss contains NaN"
        if return_each:
            return loss
        else:
            return loss.mean()


def get_random_mask(B, T, mask_prob):
    """
    生成随机 mask，表示哪些时间步要被 masked。
    Args:
        B (int): batch size
        T (int): 序列长度
        mask_prob (float): 每个时间步被 mask 的概率（0~1）

    Returns:
        mask (torch.Tensor): shape (B, T)，其中 1 表示被 mask，0 表示未被 mask
    """
    # 从均匀分布采样，并与 mask_prob 比较得到布尔 mask
    mask = torch.rand(B, T) < mask_prob  # (B, T)，bool 类型
    return mask.int()  # 转成 int 类型，1 表示 mask，0 表示保留


# TransformerEncoderLayer + AttentionPool
class FixShapeEncoder(nn.Module):
    def __init__(self,num_joints,d_model,joint_size,
                 window_len,
                 abs_pos_encoding,num_layers,
                 dropout_rate,
                 num_head_spacial, num_head_temporal,
                 shared_templ_kv,
                 temp_abs_pos_encoding,temp_rel_pos_encoding,
                 use_posteriors,
                 posterior_name,num_heads_posterior,
                 epsilon, tanh_scale,
                 dim_for_decoder,num_heads_decoder,
                 pretrain_loss_type,
                 use_attention_pool):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.window_len = window_len
        # look_ahead_mask = create_look_ahead_mask(self.window_len)
        # encoder 不必 mask
        # look_ahead_mask = torch.zeros(self.window_len, self.window_len)
        # self.register_buffer('look_ahead_mask',look_ahead_mask)
        self.abs_pos_encoding = abs_pos_encoding
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.joint_size = joint_size
        self.embeddings = nn.ModuleList([
            nn.Linear(self.joint_size,self.d_model)
            for _ in range(self.num_joints)
        ])
        pos_encoding = positional_encoding(self.window_len,self.d_model)
        self.register_buffer('pos_encoding', torch.Tensor(pos_encoding))
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.para_transformer_layers = nn.ModuleList([
            ParaTransformerLayer(d_model, num_head_spacial, num_head_temporal, 
                 num_joints, dropout_rate,
                 shared_templ_kv, temp_abs_pos_encoding,
                 window_len, temp_rel_pos_encoding,
                 use_posteriors, posterior_name,
                 num_heads_posterior,
                 epsilon, tanh_scale,) # use_posteriors: False
            for _ in range(self.num_layers)
        ])
        
        dim_out = dim_for_decoder
        
        self.pretrain_output = nn.Linear(self.d_model,joint_size)
        self.loss_type = pretrain_loss_type
        
        self.use_attention_pool = use_attention_pool
        self.pointnet = nn.Linear(self.d_model,dim_out) # dim_out < (d_model * num_joints)
        if use_attention_pool:
            num_heads = num_heads_decoder
            depth = self.d_model // num_heads
            C = depth * (depth - 1) / 2
            C = int(C)
            self.pool = AttentionPool(cls_num=C, d_model=dim_out)
            # (B, )
            self.last_layernorm = nn.LayerNorm(dim_out)
    def pretrain_forward(self, inputs, mask_prob = 0.3, mask=None):
        # inputs: (B, T, N, dim_in), T较高
        # return: context (B, T, N, D)
        # encode each rotation matrix to the feature space (d_model)
        # different joints have different encoding matrices
        B,T,N,d = inputs.shape
        inputs = inputs.permute(2,0,1,3) # (n,b,seq,joint_size)
        embed = []
        for joint_idx in range(self.num_joints):
            # [(batch_size, seq_len, d_model)]
            joint_rep = self.embeddings[joint_idx](inputs[joint_idx])
            embed += [joint_rep]
        x = torch.stack(embed) # (n,b,seq_len,d_model)
        x = x.permute(1,2,0,3)
        # add the positional encoding
        inp_seq_len = inputs.shape[2]
        if self.abs_pos_encoding:
            x += self.pos_encoding[:, :inp_seq_len]
        
        x = self.input_dropout(x)
        # put into several attention layers
        # (batch_size, seq_len, num_joints, d_model)
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}

        # random_mask: (inp_seq_len, inp_seq_len)
        # 预测为1的位置
        random_mask = get_random_mask(B,T,mask_prob) if mask is None else mask # return: (B, T)
        random_mask = random_mask.to(x.device)
        for i in range(self.num_layers):
            x, block1, block2 = self.para_transformer_layers[i](x,random_mask,context=None,mask_type='random')
            attention_weights_temporal += [block1]  # (batch_size, num_joints, num_heads, seq_len)
            attention_weights_spatial += [block2]  # (batch_size, num_heads, num_joints, num_joints)
        # x (B,T,N,D)
        attention_weights['temporal'] = torch.stack(attention_weights_temporal, axis=1)  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        attention_weights['spatial'] = torch.stack(attention_weights_spatial, axis=1)  # (batch_size, num_layers, num_heads, num_joints, num_joints)
        return x, attention_weights, random_mask

    def forward(self, x):
        # x: (B, T, N, dim_in), T较高
        # return: context (B, T, N, D)
        # encode each rotation matrix to the feature space (d_model)
        # different joints have different encoding matrices
        x, attention_weights, _ = self.pretrain_forward(x, 0) # (B,T,N,D)
        x = self.pointnet(x) # (B,T,N,D_dec)
        if self.use_attention_pool:
            x = x.max(dim=2).values # (B, T, D_dec)
            # 注意力池化
            context = self.pool(x) # (B, d*(d-1)/2, D_dec)
            #  d*(d-1)/2 是构建后验旋转参数的维度
            context = self.last_layernorm(context)
        else:
            context = x
        return context, attention_weights
    
    def get_pretrain_loss(self, inputs, mask_prob=0.3, return_each=False):
        # inputs: (B, T, N, dim_in)
        x, _, random_mask = self.pretrain_forward(inputs, mask_prob)
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        # random_mask: (B, T)
        all_b, all_t = torch.nonzero(random_mask, as_tuple=True)  # 得到所有 (b, t)
        x_masked = x[all_b,all_t]
        # x_masked: (?, N, D)
        output = self.pretrain_output(x_masked)
        targets = inputs[all_b,all_t]
        # (?,N,d)
        seq_len = T
        diff = targets - output # (?,N,d)
        if self.loss_type == 'all_mean':
            loss = (diff ** 2).mean(dim=(1, 2))
            # (B,)
        elif self.loss_type == 'joint_sum':
            per_joint_loss = diff.view(-1, self.num_joints, self.joint_size) # (?, num_joints, joint_size)
            per_joint_loss = torch.sqrt(torch.sum(per_joint_loss ** 2, dim=-1))  # (?, num_joints)
            loss = torch.mean(per_joint_loss, dim=-1)  # (?,)
        elif self.loss_type == 'geodesic': # 测地线
            if self.joint_size == 6:
                raise ValueError('geodesic loss expect joint_size 9, got [{}],\nplease use "all_mean" or "joint_sum"'.format(self.joint_size))
            target_angles = targets.view(-1, self.num_joints, 3, 3)
            predicted_angles = output.view(-1, self.num_joints, 3, 3)
            # 转置 predicted: (?, joints, 3, 3)
            pred_transpose = predicted_angles.transpose(-1, -2)
            # 矩阵乘法: R1 * R2^T
            m = torch.matmul(target_angles, pred_transpose)  # (?, joints, 3, 3)
            # 计算 trace
            trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] # (?, joints)
            cos = (trace - 1) / 2
            # clamp 防止数值误差引起 nan
            eps = 1e-6
            cos = torch.clamp(cos, -1.0+eps, 1.0-eps)
            theta = torch.acos(cos)  # (batch, seq_len, joints)
            loss = torch.sum(theta, dim=-1)  # (?,)
        else:
            raise ValueError("unknown loss_type")
        assert not torch.isnan(loss).any(), "loss contains NaN"
        if return_each:
            return loss
        else:
            return loss.mean()
        

def sample_tau(batch_size, time_steps, Tau, device=None, mode='uniform'):
    """
    从 [1, Tau] 中采样 (B, T) 的 tau 值。
    
    参数:
        batch_size: int, B
        time_steps: int, T
        Tau: int, 扩散时间上限
        device: torch.device, 默认和模型保持一致
        mode: 采样模式，目前支持 'uniform' 均匀分布
    
    返回:
        tau: (B, T) int 类型张量
    """
    if mode == 'uniform':
        tau = torch.randint(low=1, high=Tau + 1, size=(batch_size, time_steps), device=device)
        return tau
    else:
        raise NotImplementedError(f"Unsupported sampling mode: {mode}")


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

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

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


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret



class PointwiseNet(nn.Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+3), # 这个3是time_emb,并非3维空间
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, point_dim, context_dim+3)
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
        for i, layer in enumerate(self.layers):
            # (B,N,d)->(B,N,d)
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out



class DiffusionPoint(nn.Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net.decode(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        # context: (B,D)
        # return: (B,N,d)
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net.decode(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]



class SpacialAttentionEncoder(nn.Module):
    def __init__(self,d_model,num_joints,num_heads_spacial,
                 use_cls_token):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints if not use_cls_token else num_joints+1
        self.num_heads_spacial = num_heads_spacial
        assert d_model%num_heads_spacial == 0, "d_model必须被num_head_spacial整除，接受了D={},H={}".format(d_model,num_heads_spacial)
        self.linear_key = nn.Linear(self.d_model,self.d_model)
        self.linear_value = nn.Linear(self.d_model,self.d_model)
        # linear_query的bias蕴含的信息多于joint_embed，所以不用加joint_embed
        self.linear_query = nn.ModuleList([
            nn.Linear(self.d_model,self.d_model)
            for _ in range(self.num_joints)
        ])
        self.linear_output = nn.Linear(self.d_model,self.d_model)
    def forward(self,x, mask=None):
        # x: (batch_size, seq_len, num_joints, d_model)
        # mask: None
        k = self.linear_key(x)
        v = self.linear_value(x)
        x = x.permute(2,0,1,3) # (N,B,L,D)
        q_joints = []
        for joint_idx in range(self.num_joints):
            q = self.linear_query[joint_idx](x[joint_idx]) # (B,L,D)
            q = q.unsqueeze(2) # (B,L,1,D)
            q_joints += [q]
        q_joints = torch.cat(q_joints,dim=2) # (B,L,N,D)
        batch_size,seq_len,_,_ = q_joints.shape
        # (batch_size,seq_len,num_heads,num_joints,depth) where depth = d_model / num_heads
        q_joints = split_heads(q_joints, batch_size, seq_len,
                            self.num_joints, self.num_heads_spacial,self.d_model)
        k = split_heads(k, batch_size, seq_len,
                        self.num_joints, self.num_heads_spacial,self.d_model)
        v = split_heads(v, batch_size, seq_len,
                        self.num_joints, self.num_heads_spacial,self.d_model)
        scaled_attention, attention_weights = scaled_dot_product_attention(q_joints, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 1, 3, 2, 4)
        concat_attention = torch.reshape(scaled_attention, (batch_size, seq_len, self.num_joints, self.d_model))
        output = self.linear_output(concat_attention)
        attention_weights = attention_weights[:, -1, :, :, :]  # (batch_size, num_heads, num_joints, num_joints)
        return output, attention_weights


class SpatialTransformer(nn.Module):
    def __init__(self,d_in,num_joints,
                 num_heads_spacial,d_model,num_layers,
                 as_decoder):
        super().__init__()
        self.num_joints = num_joints
        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.joint_embed = nn.ModuleList([
            nn.Linear(d_in,d_model)
            for _ in range(num_joints)
        ])
        self.attns = nn.ModuleList([
            SpacialAttentionEncoder(d_model,num_joints,
                                    num_heads_spacial,
                                    use_cls_token=True)
            for _ in range(num_layers)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        if as_decoder:
            self.output_proj = nn.ModuleList([
                nn.Linear(d_model, d_in)
                for _ in range(num_joints)
            ])
        self.as_decoder = as_decoder
    def forward(self,x,context=None,beta=None):
        # x: (B,N,d)
        # context: (B,D-3) or None
        # beta: (B,) or None
        # return: (B,N+1,D)
        B,N,d = x.shape
        output = []
        for i in range(self.num_joints):
            joint_rep = x[:,i,:] # (B,d)
            joint_rep = self.joint_embed[i](joint_rep) # (B,D)
            output += [joint_rep]
        # [(B,D)]
        output = torch.stack(output,dim=1) # (B,N,D)
        # add cls token
        if context is None:
            cls_token = self.cls_token # (D)
            cls_token = cls_token[None,None,:].expand(B,-1,-1) # (B,1,D)
        else:
            assert beta is not None, "decode必须设置beta"
            tau_emb = torch.stack([beta, torch.sin(beta), torch.cos(beta), torch.cos(beta/2)], dim=-1)  # (B, 3)
            cls_token = torch.cat([context,tau_emb],dim=-1) # (B,D)
            cls_token = cls_token[:,None,:] # (B,1,D)
        output = torch.cat([cls_token, output], dim=1)
        # 经过多个spatial attention层
        inputs = output[:,None,:,:] # (B,1,N+1,D)
        for l in range(self.num_layers):
            output, _ = self.attns[l](inputs) # (B,1,N+1,D)
            inputs = self.lns[l](inputs + output)
        return inputs[:,0,:,:] # (B,N+1,D)
    def encode(self,x):
        # (B,N,d)
        # return: (B,D)
        assert (~self.as_decoder), "必须选择as_decoder才能调用decode方法"
        output = self(x) # (B,N+1,D)
        return output[:,0,:]
    def decode(self,x,beta,context):
        assert self.as_decoder, "必须选择as_decoder才能调用decode方法"
        # x:(B,N,d)
        # beta:(B,)
        # context: (B,D) 包含信息: tau和context
        # return: (B,N,d)
        rep = self(x,context,beta) # (B,N,D)
        outputs = []
        for j in range(self.num_joints):
            output = self.output_proj[j](rep[:,j,:]) # (B,d)
            outputs += [output]
        # [(B,d)]
        outputs = torch.stack(outputs,dim=1) # (B,N,d)
        return x + outputs # decode

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_joints,
                 num_spatial_heads, num_encoder_layers,
                 num_steps=500,beta_1=1e-4,beta_T=0.02,
                 sched_mode='linear'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # self.encoder = PointNetEncoder(zdim=latent_dim,input_dim=input_dim)
        self.encoder = SpatialTransformer(d_in=input_dim,num_joints=num_joints,
                                      num_heads_spacial=num_spatial_heads,
                                      d_model=latent_dim-4,num_layers=num_encoder_layers,
                                      as_decoder=False) # 真正的嵌入维度是latent_dim-4
        self.diffusion = DiffusionPoint(
            net = SpatialTransformer(d_in=input_dim,num_joints=num_joints,
                                      num_heads_spacial=num_spatial_heads,
                                      d_model=latent_dim,num_layers=num_encoder_layers,
                                      as_decoder=True),
            var_sched = VarianceSchedule(
                num_steps=num_steps,
                beta_1=beta_1,
                beta_T=beta_T,
                mode=sched_mode
            )
        )
    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code = self.encoder.encode(x)# 没有用 sigma ?
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, point_dim=self.input_dim, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x):
        # x: (B, N, d)
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code)
        return loss

class FrameRepNet(nn.Module):
    def __init__(self,d_in,d_out,joint_num,
                 num_spatial_heads,num_encoder_layers,
                 num_steps,beta_1,beta_T,sched_mode):
        super().__init__()
        self.d_out = d_out
        self.joint_num = joint_num
        self.joint_size = d_in
        self.vae = AutoEncoder(input_dim=d_in,
                               latent_dim=d_out,
                               num_joints=joint_num,
                               num_spatial_heads=num_spatial_heads,
                               num_encoder_layers=num_encoder_layers,
                               num_steps=num_steps,
                               beta_1=beta_1,
                               beta_T=beta_T,
                               sched_mode=sched_mode
                               )
    def encode(self,poses):
        # poses:(B,T,N,d)
        # return: (B,T,D) 得到帧表示（可以逐帧还原）
        B,T,N,d = poses.shape
        D = self.d_out - 4
        poses = torch.flatten(poses,0,1) # (B*T,N,d)
        poses_rep_seq = self.vae.encode(poses) # (B*T,N,D)
        poses = torch.reshape(poses_rep_seq, (B,T,D))
        return poses
    def decode(self,poses_rep_seq):
        # 测试时使用，采样500步，耗时。
        # poses_rep_seq:(B,T,D)
        # return: (B,T,N,d) 得到还原
        N = self.joint_num
        d = self.joint_size
        B,T,D = poses_rep_seq.shape
        poses_rep_seq = torch.flatten(poses_rep_seq,0,1) # (B*T,N,d)
        poses_recons = self.vae.decode(poses_rep_seq,N) # (B*T,N,d)
        poses_recons = torch.reshape(poses_recons,(B,T,N,d))
        return poses_recons
    def get_loss(self,poses):
        # poses:(B,T,N,d)
        poses = torch.flatten(poses,0,1) # (B*T,N,d)
        loss = self.vae.get_loss(poses)
        return loss




# 只能处理点云数据，骨骼旋转角度不行
class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim, idx_num=None):
        super().__init__()
        self.zdim = zdim
        offset = 0 # input_dim - 3
        
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
        # (B,N,d)
        x = x.transpose(1, 2)
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




# 只有嵌入和一个spacial attention
class HistoryEncoder(nn.Module):
    def __init__(self,d_in,d_out,window_len,joint_num,
                 num_head_spacial,dropout_rate):
        super().__init__()
        self.window_len = window_len
        self.joint_num = joint_num
        time_encoding = positional_encoding(window_len,d_out) # (1,T,1,D)
        self.register_buffer('time_encoding', torch.Tensor(time_encoding))
        self.convs = nn.ModuleList([
            nn.Conv1d(d_in,d_out,kernel_size=3,padding=1)
            for _ in range(joint_num)
        ])
        # self.ln_temporal = nn.LayerNorm(d_out)
        self.spatial_attn = SepSpacialAttention(d_out,joint_num,num_head_spacial)
        self.dropout_spatial = nn.Dropout(dropout_rate)
        # self.ln_spatial = nn.LayerNorm(d_out)
        self.joint_embedding = nn.Parameter(torch.randn(joint_num,d_out))
    def forward(self,history_poses,start_t=0):
        # history_poses: (B,T,N,d)
        B,T,N,d = history_poses.shape
        assert N==self.joint_num, "关节数目不匹配,模型期待{},得到了输入{}".format(self.joint_num,N)
        history_poses = history_poses.permute(2,0,3,1) # (N,B,d,T)
        output = []
        # 时间卷积嵌入 (d,T) -conv1d,kernel_size=3,padding=1-> (D,T)
        for joint_idx in range(N):
            output += [self.convs[joint_idx](history_poses[joint_idx])] # (B,d,T)
        output = torch.stack(output) # (N,B,D,T)
        output = output.permute(1,3,0,2) # (B,T,N,D)
        # output = self.ln_temporal(output) # (B,T,N,D)
        output = F.relu(output) # (B,T,N,D)
        # 空间编码
        output += self.joint_embedding[None,None,:,:] # (B,T,N,D)
        # 空间 attention 聚合
        attn_spa, attn_weights_block = self.spatial_attn(output)
        attn_spa = self.dropout_spatial(attn_spa)
        # attn_spa = self.ln_spatial(attn_spa + output)
        # 时间编码
        output += self.time_encoding[:,start_t:start_t+T] # (B,T,N,D)
        return output, None # 为了匹配另一个encoder的输出格式
    def encode(self,history_poses):
        output, _ = self(history_poses)
        return output


class FlexConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding=0,stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
    def forward(self, x):
        # x: (..., in_channels, T)
        # return (..., out_channels, T_out)
        *leading_dims, in_channels, T = x.shape
        x = x.reshape(-1, in_channels, T)  # (B*, in_channels, T)
        x = self.conv(x)                   # (B*, out_channels, T_out)
        out_channels = x.shape[1]
        T_out = x.shape[2]
        x = x.view(*leading_dims, out_channels, T_out)  # (..., out_channels, T_out)
        return x

class EncodeCompressor(nn.Module):
    def __init__(self,D_decoder,joint_num,
                 d_in,window_len,num_head_spacial,dropout_rate):
        super().__init__()
        self.encoder = HistoryEncoder(d_in,D_decoder,window_len,joint_num,
                                      num_head_spacial,dropout_rate) # history_poses
        C = D_decoder * (D_decoder-1) / 2
        # self.pool = AttentionPool(cls_num=C, d_model=D_decoder, joint_num=joint_num)
        self.conv = FlexConv1d(1,C,kernel_size=15, stride=15) # 0.5 秒 30 fps
        self.post_linear = nn.ModuleList([
            nn.Linear(C,C)
            for _ in range(joint_num)
        ])
    def forward(self,history_poses):
        # (B,T,N,d)
        B,T,N,d = history_poses.shape
        context = self.encoder.encode(history_poses) # (B,T,N,D)
        # (B,T,N,D)
        context = context.permute(2,0,3,1) # (N,B,D,T)
        output = []
        for n in range(N):
            context_n = context[n] # (B,D,T)
            # 先用卷积找d个极大的idx
            context_n = context_n[:,:,None,:] # (B,D,1,T)
            context_n = self.conv(context_n) # (B,D,C,T)
            context_n, _ =torch.max(context_n, dim=-1) # (B,D,C)
            context_n = self.post_linear[n](context_n) # (B,D,C)
            output.append(context_n)
        output = torch.stack(output,dim=1) # (B,N,D,C)
        return output


class TransformerDiffusionMultiStep(nn.Module):
    def __init__(self,num_joints,joint_size,
                 window_len,
                 d_model_encoder,num_layers_encoder,dropout_rate_encoder,
                 abs_pos_encoding_encoder,temp_abs_pos_encoding_encoder,temp_rel_pos_encoding_encoder,
                 num_head_spacial_encoder,num_head_temporal_encoder,
                 shared_templ_kv_encoder,
                 pretrain_loss_type,
                 d_model_decoder, num_heads_posterior_decoder,
                 abs_pos_encoding_decoder, num_layers_decoder, dropout_rate_decoder,
                 use_6d_outputs, residual_velocity,
                 loss_type,
                 num_head_spacial_decoder,num_head_temporal_decoder,
                 shared_templ_kv_decoder,
                 temp_abs_pos_encoding_decoder,temp_rel_pos_encoding_decoder,
                 posterior_name,
                 epsilon,tanh_scale,
                 Tau, schedule,
                 pred_time_step,
                 use_simple_history_encoder,
                 need_output_bias,
                 need_normer,
                 use_posterior,
                 use_simple_posterior_encoder):
        super().__init__()
        if use_simple_posterior_encoder:
            pass
        else:
            self.encoder = FixShapeEncoder(num_joints,d_model_encoder,joint_size,
                    window_len,
                    abs_pos_encoding_encoder,num_layers_encoder,
                    dropout_rate_encoder,
                    num_head_spacial_encoder, num_head_temporal_encoder,
                    shared_templ_kv_encoder,
                    temp_abs_pos_encoding_encoder,temp_rel_pos_encoding_encoder,
                    use_posteriors=False,
                    posterior_name=None,num_heads_posterior=None,
                    epsilon=None, tanh_scale=None,
                    dim_for_decoder=d_model_decoder,num_heads_decoder=num_heads_posterior_decoder,
                    pretrain_loss_type=pretrain_loss_type,
                    use_attention_pool=True)
        if use_simple_history_encoder: # 是否使用简单的历史编码器
            self.history_encoder = HistoryEncoder(joint_size, d_model_decoder, window_len, num_joints,
                                                  num_head_spacial_encoder, dropout_rate_decoder)
        else:
            self.history_encoder = FixShapeEncoder(num_joints,d_model_encoder,joint_size,
                     window_len,
                     abs_pos_encoding_encoder,num_layers_encoder,
                     dropout_rate_encoder,
                     num_head_spacial_encoder, num_head_temporal_encoder,
                     shared_templ_kv_encoder,
                     temp_abs_pos_encoding_encoder,temp_rel_pos_encoding_encoder,
                     use_posteriors=False,
                     posterior_name=None,num_heads_posterior=None,
                     epsilon=None, tanh_scale=None,
                     dim_for_decoder=d_model_decoder,num_heads_decoder=num_heads_posterior_decoder,
                     pretrain_loss_type=pretrain_loss_type,
                     use_attention_pool=False) # (B,T,N,d_in)
        self.decoder = TransformerDecoder(num_joints, d_model_decoder, window_len,
                 abs_pos_encoding_decoder, num_layers_decoder, dropout_rate_decoder,
                 use_6d_outputs, joint_size, residual_velocity,
                 loss_type,
                 num_head_spacial_decoder,num_head_temporal_decoder,
                 shared_templ_kv_decoder,
                 temp_abs_pos_encoding_decoder,temp_rel_pos_encoding_decoder,
                 use_posteriors=use_posterior,
                 posterior_name=posterior_name,num_heads_posterior=num_heads_posterior_decoder,
                 epsilon=epsilon,tanh_scale=tanh_scale,
                 need_output_bias=need_output_bias)
        self.diffusion_schedule = DiffusionSchedule(Tau, schedule)
        tau_embedding = positional_encoding(Tau, d_model_decoder) # (1,Tau,1,D)
        tau_embedding = torch.Tensor(tau_embedding)
        self.register_buffer('tau_embedding', tau_embedding[0,:,0,:]) # (Tau,D)
        # v = rot_mat[1:] - rot_mat[:-1] 是 1e-5 量级的
        noise_scale = torch.ones((num_joints,joint_size))
        self.register_buffer('noise_scale', noise_scale)
        # self.noise_scale =  nn.Parameter(torch.randn((num_joints,joint_size)) * 0.02) # (N,d)
        # 这被证明是一个失败的设计
        
        self.pred_time_step = pred_time_step # 一次性预测(生成)的时间步数
        self.normer = Normer() if need_normer else NotNormer() # Normer: 输入的不必归一化，生成的须归一化, NotNormer:恒等映射
        
        # 外部调用方法: 
        # forward, 计算loss
        # sample, 用于生成
        
    def encode(self, inputs):
        # inputs: (B, T, N, d_in)
        # let D1=dim_encoder, D2=dim_decoder, H2=num_heads_decoder
        # let d = depth_decoder = D2 / H2
        # let C = d * (d-1) / 2
        # return: (B, C, D2)
        if inputs is None:
            return None
        context, _ = self.encoder(inputs)
        return context
    
    def history_encode(self,history_poses):
        # history_poses: (B, T, N, d_in)
        # return: (B, T, N, D2)
        prior_context, _ = self.history_encoder(history_poses)
        return prior_context
    
    def decode(self, poses_tau, prior_context, tau_emb, context, mask):
        # poses_tau: (B, T_pred, N, d_in)
        # history_poses: (B, T, N, d_in)
        # tau_emb: (B, D)
        # context: (B, C, D2)
        # return: (B, T_pred, N, d_in)
        outputs, _ = self.decoder(poses_tau, prior_context, tau_emb, context, mask)
        return outputs
    
    def forward(self, long_seq, inputs, t_pred=None, tau=None):
        # long_seq: (B, T_l, N, d_in)
        # inputs: (B, T, N, d_in)
        # tau: int 扩散模型时间
        # t_pred: 一次性预测的步数
        # outputs: float
        B, T, N, d_in = inputs.shape
        Tau = self.diffusion_schedule.Tau
        context = self.encode(long_seq) # (B, C, D2)
        tau = sample_tau(B, 1, Tau, inputs.device) if tau is None else tau # (B, 1)
        tau = tau[:,0]
        loss = self.get_loss(inputs, context, tau, t_pred)
        return loss
    
    def tau_embed(self, tau):
        # tau: (B,)
        # return: (B, D)
        B = tau.shape[0]
        tau = tau - 1 # 0-based index
        tau_embedding = self.tau_embedding[tau.view(-1)].reshape(B,-1) # (B, D)
        return tau_embedding

    def get_loss(self, inputs, context, tau, pred_time_step=None):
        # inputs: (B, T, N, d_in)
        # context: (B, C, D2)
        # tau: 扩散模型时间 (B,) {0,1,2,3,...,Tau-1}
        # pred_time_step: int 一次性预测步数
        pred_time_step = self.pred_time_step if pred_time_step is None else pred_time_step

        poses_0 = inputs[:,-pred_time_step:]
        history_poses = inputs[:,:-pred_time_step]
        T = history_poses.shape[1]
        
        self.normer.update_state(history_poses) # 更新均值方差 (B,1,N,D)
        # 别用数据增强 history_poses = self.normer.strengthen(history_poses)
        poses_0 = self.normer.norm_with_existed_state(poses_0) # 要预测的家伙不包含均值方差，因为history_poses已经包含了
        
        assert T==pred_time_step,"woc"
        
        beta_tau = self.diffusion_schedule.betas[tau] # (B,)
        alpha_tau = self.diffusion_schedule.alphas[tau] # (B,)
        alpha_bar_tau = self.diffusion_schedule.alpha_bars[tau] # (B,)

        B, T_pred, N, D = poses_0.shape
        epsilon = torch.randn_like(poses_0)
        # 加噪
        sqrt_alpha_bar = alpha_bar_tau.sqrt().view(B, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - alpha_bar_tau).sqrt().view(B, 1, 1, 1)
        # 缩放到同一水平
        epsilon_scaled = epsilon * self.noise_scale[None,None,:,:]
        poses_tau = poses_0 * sqrt_alpha_bar + epsilon_scaled * sqrt_one_minus_alpha_bar  # (B, T_pred, N, D)

        # tau: (B)
        tau_embed = self.tau_embed(tau) # (B, D)
        
        mask = torch.zeros((T_pred,T)).to(inputs.device) # 没有mask
        
        # history_poses = self.normer.norm_with_existed_state(history_poses)
        # history_poses = self.normer._strengthen_noise(history_poses) # 数据增强
        # history_poses = self.normer.denorm(history_poses)
        
        prior_context = self.history_encode(history_poses) # (B,T,N,D)
        
        epsilon_pred_scaled = self.decode(poses_tau, prior_context, tau_embed, context, mask) # (B, T, N, d_in)
        epsilon_pred = epsilon_pred_scaled / self.noise_scale[None,None,:,:]
        loss = ((epsilon_pred - epsilon)**2).mean()
        self.normer.clear_state() # 清除状态，以防记忆混淆
        return loss

    # 以下是推理相关
    def denoise(self, poses_tau, prior_context, context, tau, sampled_z=None):
        # 确保模型处于eval模式
        # poses_tau: (B, T_q, N, d_in) , 一般 T_q 等于 self.pred_time_step
        # tau: int 扩散模型时间
        # prior_context: (B, T, N, D), 实际上是 prior_context
        # context: (B, C, D2)
        # sampled_z: (B, T_q, N, d_in)
        # return poses_tau_minus_one: (B, T_q, N, d_in) 去除一步噪声
        B, T, N, D = prior_context.shape
        T_q = poses_tau.shape[1]

        beta_tau = self.diffusion_schedule.betas[tau]
        alpha_tau = self.diffusion_schedule.alphas[tau]
        alpha_bar_tau = self.diffusion_schedule.alpha_bars[tau]
        
        mask = torch.zeros(T_q,T).to(poses_tau.device) # cross attention mask
        
        # tau: (B, T_q)
        tau_tensor = torch.full((B,),tau,dtype=torch.long,device=poses_tau.device)
        tau_embed = self.tau_embed(tau_tensor) # (B, D)
        
        epsilon_scaled_theta = self.decode(poses_tau, prior_context, tau_embed, context, mask)
        # (B, T_q, N, d_in)
        _, T_q, _, d = poses_tau.shape
        mu_tau = 1/alpha_tau.sqrt() * (poses_tau - beta_tau/(1-alpha_bar_tau).sqrt() * epsilon_scaled_theta)
        if tau == 1:
            return mu_tau
        alpha_bar_tau_minus_one = self.diffusion_schedule.alpha_bars[tau-1]
        var_tau = beta_tau * (1-alpha_bar_tau_minus_one) / (1-alpha_bar_tau)
        sigma_tau = var_tau.sqrt()
        sampled_z = torch.randn((B, 1, N, d),device=mu_tau.device) if sampled_z is None else sampled_z
        sampled_z *= 1 #*self.noise_scale[None,None,:,:] #
        poses_tau_minus_one = mu_tau + sigma_tau * sampled_z
        return poses_tau_minus_one

    def sample(self, history_poses, context, sampled_z=None, pred_time_step=None):
        # history_poses: (B, T, N, d_in)
        # context: (B, C, D2)
        pred_time_step = self.pred_time_step if pred_time_step is None else pred_time_step
        B, T, N, d_in = history_poses.shape
        
        self.normer.update_state(history_poses) # 更新状态
        # 初始噪声
        poses_tau = torch.randn(B, pred_time_step, N, d_in).to(history_poses.device)
        # 噪声水平控制
        poses_tau *= self.noise_scale[None,None,:,:]
        # 开始去噪
        Tau = self.diffusion_schedule.Tau # cosine选50步可以吗
        # 计算 prior_context
        prior_context = self.history_encode(history_poses)
        for tau in range(Tau,0,-1): # [Tau, Tau-1,..., 1]
            poses_tau = self.denoise(poses_tau, prior_context, context, tau, sampled_z=sampled_z)
        poses_tau = self.normer.denorm(poses_tau)
        self.normer.clear_state()
        return poses_tau # (B, T_q, N, d_in)

    def get_pretrain_params(self):
        # 无监督学习，mask_token预测
        # 获取 prediction_model, estimator.zl_embedding, estimator.post_enc 的参数给optim
        pred_params = list(self.encoder.parameters())
        return pred_params

    def get_history_encoder_pretrain_loss(self, history_poses, p_mask=0.3):
        # history_poses (B,T,N,d)
        # strength_history_poses = self.normer.strengthen(history_poses, p=p_mask, return_mask=False)
        loss = self.history_encoder.get_pretrain_loss(history_poses, mask_prob=p_mask)
        return loss
    
    def get_history_encoder_params(self):
        pred_params = list(self.history_encoder.parameters())
        return pred_params

    def get_pretrain_loss(self, long_seq, mask_prob):
        # long_seq: (B, T_l, N, d_in)
        loss = self.encoder.get_pretrain_loss(long_seq, mask_prob)
        return loss

    def get_train_params(self):
        # 获取训练模型参数
        pred_params = list(self.encoder.parameters())
        pred_params += list(self.decoder.parameters())
        pred_params += list(self.history_encoder.parameters())
        # pred_params += list(self.tau_embedding)
        # pred_params += [self.noise_scale]
        return pred_params

    def get_train_params_without_encoder(self):
        # 获取训练模型参数
        pred_params = list(self.decoder.parameters())
        # pred_params = list(self.encoder.parameters())
        # pred_params += list(self.history_encoder.parameters())
        # pred_params += list(self.tau_embedding)
        # pred_params += [self.noise_scale]
        return pred_params

# 作为Normer的对比
class NotNormer:
    def __init__(self):
        pass
    def norm(self,inputs):
        # inputs (B,T,N,d)
        return inputs
    def denorm(self,inputs_normed):
        # inputs_normed (B,T1,N,d)
        return inputs_normed
    def update_state(self,inputs):
        # inputs (B,T,N,d)
        pass
    def clear_state(self):
        pass
    def norm_with_existed_state(self,inputs):
        return inputs

    

class Normer:
    def __init__(self):
        # self.sha256 = None
        self.mu = None
        self.sigma = None
    def strengthen(self,inputs,p=0.1,return_mask=False):
        # inputs (B,T,N,d)
        mask = self._strengthen_mask(inputs)
        inputs = self._strengthen_noise(inputs)
        if return_mask:
            return inputs, mask
        else:
            return inputs * (~mask)
    def _strengthen_mask(self,inputs,mask_prob=0.05):
        # inputs: (B, T, N, d)
        mask = (torch.rand_like(inputs, device=inputs.device) < mask_prob) # (B,T,N,d)
        return mask
    def _strengthen_noise(self, inputs, noise_level=0.05):
        # 高斯噪声
        noise = torch.randn_like(inputs, device=inputs.device) * noise_level
        return inputs + noise
    def norm(self,inputs):
        # inputs (B,T,N,d)
        self.mu = torch.mean(inputs,dim=1,keepdim=True) # (B,1,N,d)
        self.sigma = torch.std(inputs,dim=1,keepdim=True)
        return (inputs - self.mu) / (self.sigma + 1e-8)
    def denorm(self,inputs_normed):
        # inputs_normed (B,T1,N,d)
        return inputs_normed * self.sigma + self.mu
    def update_state(self,inputs):
        # inputs (B,T,N,d)
        self.mu = torch.mean(inputs,dim=1,keepdim=True) # (B,1,N,d)
        self.sigma = torch.std(inputs,dim=1,keepdim=True)
        # 更新状态
    def clear_state(self):
        self.mu = None
        self.sigma = None
    def norm_with_existed_state(self,inputs):
        return (inputs - self.mu) / (self.sigma + 1e-8)


# (B, C, D2) -> (B, C, D2)
# 隐层更新
# 之后再说吧
class LatentMotionModel(nn.Module):
    def __init__(self, latent_c, dim_rep):
        super().__init__()
        # 让(B, C, D2)->(B, C1, D2)
        # 也许可以用C1个可学习的query去交叉注意(B, C, D2)
        # 得到的(B, C1, D2) 再通过一个自注意力层模拟动力学演化
        # 最后再以之前的(B, C, D2)为query得到最终的(B, C, D2)
        # 也就是演化后的context
        self.query_slots = nn.Parameter(torch.randn(1, latent_c, dim_rep))
        
    def forward(self, context, stimulus=None):
        # context: (B, C, D2) 心理表示
        # stimulus: 外界对心理表示的作用力
        # stimulus为None时，心理表示依惯性自主演化
        # (B, C, D2)->(B, C1, D2)->(B, C, D2)
        # 因为C维度是后验旋转参数，D2维度是它的特征
        # 我们希望在C维度上拟合一个动力学过程
        # context 一次向后预测 T 个时刻
        pass


# mask_block允许模型一次预测多步

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
# loss = cal_KL(mu_z_s, log_sigma_z_s, mu_z_l, log_sigma_z_l )