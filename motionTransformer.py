# 让context决定一个微小旋转，作用在attention层的query向量上，
# 这样context为None时就是先验预测，context存在时就是后验预测。
# 问题是，如何表示d_model维的一个微小旋转

# 任意旋转矩阵都可以由一个反对称矩阵的矩阵指数构造

# 之前的问题在于，模型过于关注context

# 从 context 构造一个 d_model × d_model 的反对称矩阵 A(c)
# 使用泰勒展开近似其指数 R ≈ I + A（微小旋转时足够）
# 得到变换后的 query：q' = (I + A) q

# 用全序列通过一个TransformerEncoder得到 context_{l for 1},context_{l for 2},...,context{l for l}

# 当第 i 步预测 i+1 步时, 用 x_{i+1} 构造后验旋转.

# layernorm后输出的是正态分布吗? (就单个样本单个特征的各个时间来说）



import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np



import sys
sys.path.append('/home/vipuser/DL/Dataset100G/DL-3D-Upload/sutra/tensorFlow-project/motion-transformer')
from common.constants import Constants as C
# from common.conversions import compute_rotation_matrix_from_ortho6d



'''
# batch*n
def normalize_vector(v, return_mag=False):
    batch = tf.shape(v)[0]
    v_mag = tf.sqrt(tf.reduce_sum(tf.pow(v, 2), axis=-1))  # batch
    v_mag = tf.maximum(v_mag, 1e-6)
    v_mag = tf.reshape(v_mag, (batch, 1))
    v_mag = tf.tile(v_mag, (1, tf.shape(v)[1]))
    v = v/v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v
'''

def normalize_vector(v, return_mag=False):
    # v: (batch_size, n)
    v_mag = torch.norm(v, dim=-1, keepdim=True)  # (batch_size, 1)
    v_mag = torch.clamp(v_mag, min=1e-6)
    v_normalized = v / v_mag
    if return_mag:
        return v_normalized, v_mag.squeeze(-1)
    else:
        return v_normalized

'''
# u, v batch*n
def cross_product(u, v):
    i = tf.expand_dims(u[:, 1]*v[:, 2] - u[:, 2]*v[:, 1], axis=-1)
    j = tf.expand_dims(u[:, 2]*v[:, 0] - u[:, 0]*v[:, 2], axis=-1)
    k = tf.expand_dims(u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0], axis=-1)
    out = tf.concat([i, j, k], axis=1)  # batch*3
    return out
'''

def cross_product(u, v):
    # u, v: (batch_size, 3)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.stack([i, j, k], dim=1)  # (batch_size, 3)
    return out

'''
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3
    
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
    
    matrix = tf.concat([tf.expand_dims(x, axis=-1),
                        tf.expand_dims(y, axis=-1),
                        tf.expand_dims(z, axis=-1)], axis=-1)  # batch*3*3
    return matrix
'''

def compute_rotation_matrix_from_ortho6d(ortho6d):
    # ortho6d: (batch_size, 6)
    x_raw = ortho6d[:, 0:3]  # (batch_size, 3)
    y_raw = ortho6d[:, 3:6]  # (batch_size, 3)

    x = normalize_vector(x_raw)           # (batch_size, 3)
    z = normalize_vector(cross_product(x, y_raw))  # (batch_size, 3)
    y = cross_product(z, x)               # (batch_size, 3)

    x = x.unsqueeze(-1)  # (batch_size, 3, 1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    matrix = torch.cat([x, y, z], dim=-1)  # (batch_size, 3, 3)
    return matrix


def scaled_dot_product_attention(q, k, v, mask, rel_key_emb=None, rel_val_emb=None):
    # attn_dim: num_joints for spatial and seq_len for temporal
    '''
    The scaled dot product attention mechanism introduced in the Transformer
    :param q: the query vectors matrix (..., attn_dim, d_model/num_heads)
    :param k: the key vector matrix (..., attn_dim, d_model/num_heads)
    :param v: the value vector matrix (..., attn_dim, d_model/num_heads)
    :param mask: a mask for attention
    :return: the updated encoding and the attention weights matrix
    '''
    # matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., num_heads, attn_dim, attn_dim)
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))

    # batch_size = tf.shape(q)[0]
    # heads = tf.shape(q)[1]
    # length = tf.shape(q)[2]
    batch_size = q.shape[0]
    heads = q.shape[1]
    length = q.shape[2]
    
    if rel_key_emb is not None:
        # q_t = tf.transpose(q, [2, 0, 1, 3])
        q_t = q.permute(2,0,1,3)
        # q_t_r = tf.reshape(q_t, [length, heads*batch_size, -1])
        q_t_r = torch.reshape(q_t, (length, heads*batch_size, -1))
        # q_tz_matmul = tf.matmul(q_t_r, rel_key_emb, transpose_b=True)
        q_tz_matmul = torch.matmul(q_t_r, rel_key_emb.transpose(-1,-2))
        # q_tz_matmul_r = tf.reshape(q_tz_matmul, [length, batch_size, heads, -1])
        q_tz_matmul_r = torch.reshape(q_tz_matmul, (length, batch_size, heads, -1))
        # q_tz_matmul_r_t = tf.transpose(q_tz_matmul_r, [1, 2, 0, 3])
        q_tz_matmul_r_t = q_tz_matmul_r.permute((1,2,0,3))
        matmul_qk += q_tz_matmul_r_t

    # scale matmul_qk
    # dk = tf.cast(tf.shape(k)[-1], tf.float32)
    dk = k.shape[-1] # int
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask.to(dtype=scaled_attention_logits.dtype) * -1e9)

    # normalized on the last axis (seq_len_k) so that the scores add up to 1.
    # attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., num_heads, attn_dim, attn_dim)
    attention_weights = F.softmax(scaled_attention_logits, -1)

    ## attention_weights = tf.layers.dropout(attention_weights, training=is_training, rate=0.2)
    # output = tf.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)
    output = torch.matmul(attention_weights, v)
    
    if rel_val_emb is not None:
        # w_t = tf.transpose(attention_weights, [2, 0, 1, 3])
        w_t = attention_weights.permute((2,0,1,3))
        # w_t_r = tf.reshape(w_t, [length, heads*batch_size, -1])
        w_t_r = torch.reshape(w_t, (length, heads*batch_size, -1))
        # w_tz_matmul = tf.matmul(w_t_r, rel_val_emb, transpose_b=False)
        w_tz_matmul = torch.matmul(w_t_r, rel_val_emb)
        # w_tz_matmul_r = tf.reshape(w_tz_matmul, [length, batch_size, heads, -1])
        w_tz_matmul_r = torch.reshape(w_tz_matmul, (length, batch_size, heads, -1))
        # w_tz_matmul_r_t = tf.transpose(w_tz_matmul_r, [1, 2, 0, 3])
        w_tz_matmul_r_t = w_tz_matmul_r.permute((1,2,0,3))
        # output += w_tz_matmul_r_t
        output += w_tz_matmul_r_t

    return output, attention_weights



def get_angles(pos, i, d_model):
    '''
    calculate the angles givin postion and i for the positional encoding formula
    :param pos: pos in the formula
    :param i: i in the formula
    :return: angle rad
    '''
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(window_len,d_model):
    '''
    calculate the positional encoding given the window length
    :return: positional encoding (1, window_length, 1, d_model)
    '''
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


def split_heads(x, shape0, shape1, attn_dim, num_heads, d_model):
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
    x = torch.reshape(x, (shape0, shape1, attn_dim, num_heads, depth))
    # return tf.transpose(x, perm=[0, 1, 3, 2, 4])
    return x.permute(0, 1, 3, 2, 4)


class SepTemporalAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model=config['transformer_d_model']
        self.num_heads=config['transformer_num_heads_temporal']
        self.shared_templ_kv=config['shared_templ_kv']
        self.temp_abs_pos_encoding=config['temp_abs_pos_encoding']
        self.window_len=config['transformer_window_length']
        self.num_joints=config['num_joints']
        pos_encoding = positional_encoding(self.window_len,self.d_model)
        self.pos_encoding = torch.Tensor(pos_encoding)
        self.temp_rel_pos_encoding = config['temp_rel_pos_encoding']
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
    def forward(self,x,mask):
        '''
        x: (batch_size, seq_len, num_joints, d_model)
        mask: temporal mask (usually the look ahead mask)
        return: the output (batch_size, seq_len, num_joints, d_model)
        '''
        if self.temp_abs_pos_encoding:
            inp_seq_len = x.shape[1]
            x += self.pos_encoding[:, :inp_seq_len]
        outputs = []
        attn_weights = []
        batch_size,seq_len,num_joints,d_model = x.shape
        x = x.permute(2,0,1,3)
        if self.shared_templ_kv:
            k_all = self.linear_k_all(x) # nn.Linear
            v_all = self.linear_v_all(x)
        rel_key_emb, rel_val_emb = None, None
        if self.temp_rel_pos_encoding:
            rel_key_emb, rel_val_emb = get_relative_embeddings(seq_len, seq_len)
        # different joints have different embedding matrices
        for joint_idx in range(self.num_joints):
            joint_rep = x[joint_idx]  # (batch_size, seq_len, d_model)
            # embed it to query, key and value vectors
            # (batch_size, seq_len, d_model)
            q = self.linear_q[joint_idx](joint_rep)
            if self.shared_templ_kv:
                v = v_all[joint_idx]
                k = k_all[joint_idx]
            else:
                # with tf.variable_scope('_key', reuse=self.reuse):
                #     k = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                # with tf.variable_scope('_value', reuse=self.reuse):
                #     v = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                k = self.linear_k[joint_idx](joint_rep)
                v = self.linear_v[joint_idx](joint_rep)
            # split it to several attention heads
            q = sep_split_heads(q, batch_size, seq_len, self.num_heads, self.d_model)
            # (batch_size, num_heads, seq_len, depth)
            k = sep_split_heads(k, batch_size, seq_len, self.num_heads, self.d_model)
            # (batch_size, num_heads, seq_len, depth)
            v = sep_split_heads(v, batch_size, seq_len, self.num_heads, self.d_model)
            # (batch_size, num_heads, seq_len, depth)
            # calculate the updated encoding by scaled dot product attention
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, rel_key_emb, rel_val_emb)
            # (batch_size, num_heads, seq_len, depth)
            # scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            scaled_attention = scaled_attention.permute(0, 2, 1, 3)
            # (batch_size, seq_len, num_heads, depth)
            # concatenate the outputs from different heads
            # concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, self.d_model])
            concat_attention = torch.reshape(scaled_attention, (batch_size, seq_len, d_model))
            # (batch_size, seq_len, d_model)
            # go through a fully connected layer
            # with tf.variable_scope(scope + '_output_dense', reuse=self.reuse):
            #     output = tf.expand_dims(tf.layers.dense(concat_attention, self.d_model), axis=2)
            output = self.linear_output(concat_attention)
            # (batch_size, seq_len, 1, d_model)
            outputs += [output.unsqueeze(2)]
            # 只取最后一个注意力层的weight
            last_attention_weights = attention_weights[:, :, -1, :]  # (batch_size, num_heads, seq_len)
            attn_weights += [last_attention_weights]
        # 拼接 num_joints
        outputs = torch.cat(outputs,dim=2)  # (batch_size, seq_len, num_joints, d_model)
        attn_weights = torch.stack(attn_weights,dim=1)  # (batch_size, num_joints, num_heads, seq_len)
        return outputs, attn_weights



class SepSpacialAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model = config['transformer_d_model']
        self.num_joints = config['num_joints']
        self.num_heads_spacial = config['transformer_num_heads_spacial']
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


class PointWiseFeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model=config['transformer_d_model']
        self.num_joints=config['num_joints']
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


class ParaTransformerLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model = config['transformer_d_model']
        self.num_head_spacial = config['transformer_num_heads_spacial']
        self.num_head_temporal = config['transformer_num_heads_temporal']
        self.num_joints = config['num_joints']
        self.dropout_rate = config['transformer_dropout_rate']
        self.temporal_attn = SepTemporalAttention(config)
        self.spatial_attn = SepSpacialAttention(config)
        self.feed_forward = PointWiseFeedForward(config)

        self.dropout_temporal = nn.Dropout(self.dropout_rate)
        self.dropout_spatial = nn.Dropout(self.dropout_rate)
        self.dropout_ff = nn.Dropout(self.dropout_rate)

        self.ln_temporal = nn.LayerNorm(self.d_model)
        self.ln_spatial = nn.LayerNorm(self.d_model)
        self.ln_ff = nn.LayerNorm(self.d_model)

    def forward(self, x, look_ahead_mask):
        # x : (batch_size, seq_len, num_joints, d_model)
        # temporal
        # print(x.shape)
        # print(look_ahead_mask.shape)
        attn1, attn_weights_block1 = self.temporal_attn(x, look_ahead_mask)
        attn1 = self.dropout_temporal(attn1)
        temporal_out = self.ln_temporal(attn1 + x)
        # spatial
        attn2, attn_weights_block2 = self.spatial_attn(x)
        attn2 = self.dropout_spatial(attn2)
        spatial_out = self.ln_spatial(attn2 + x)

        out = temporal_out + spatial_out

        ffn_output = self.feed_forward(out)
        ffn_output = self.dropout_ff(ffn_output)
        final = self.ln_ff(ffn_output + out)

        return final, attn_weights_block1, attn_weights_block2



def create_look_ahead_mask(window_len):
    # shape: (window_len, window_len)
    # return: 下三角
    return torch.triu(torch.ones(window_len, window_len), diagonal=1)


class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.num_joints = config['num_joints']
        self.d_model = config['transformer_d_model']
        
        self.window_len = config['transformer_window_length']
        look_ahead_mask = create_look_ahead_mask(self.window_len)
        self.register_buffer('look_ahead_mask',look_ahead_mask)
        self.abs_pos_encoding = config['abs_pos_encoding']
        self.num_layers = config['transformer_num_layers']
        
        self.dropout_rate = config['transformer_dropout_rate']
        self.use_6d_outputs = config['use_6d_outputs']
        self.joint_size = config['joint_size']
        
        self.residual_velocity = config['residual_velocity']
        self.loss_type = config['loss_type']
        
        self.embeddings = nn.ModuleList([
            nn.Linear(self.joint_size,self.d_model)
            for _ in range(self.num_joints)
        ])
        pos_encoding = positional_encoding(self.window_len,self.d_model)
        self.register_buffer('pos_encoding', torch.Tensor(pos_encoding))
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.para_transformer_layers = nn.ModuleList([
            ParaTransformerLayer(config)
            for _ in range(self.num_layers)
        ])
        if not self.use_6d_outputs:
            self.output_linears = nn.ModuleList([
                nn.Linear(self.d_model,self.joint_size)
                for _ in range(self.num_joints)
            ])
        else:
            self.output_linears = nn.ModuleList([
                nn.Linear(self.d_model,6)
                for _ in range(self.num_joints)
            ])
            
    def forward(self, inputs):
        '''
        The attention blocks
        :param inputs: inputs (batch_size, seq_len, num_joints, joint_size)
        :param look_ahead_mask: the look ahead mask (seq_len,seq_len), 谁能看到谁
        :return: outputs (batch_size, seq_len, num_joints, joint_size) 各自向后预测一步
        '''
        # encode each rotation matrix to the feature space (d_model)
        # different joints have different encoding matrices
        # inputs = tf.transpose(inputs, [2, 0, 1, 3])  # (num_joints, batch_size, seq_len, joint_size)
        inputs = inputs.permute(2,0,1,3) # (n,b,seq,joint_size)
        embed = []
        for joint_idx in range(self.num_joints):
            # with tf.variable_scope("embedding_" + str(joint_idx), reuse=self.reuse):
            #     joint_rep = tf.layers.dense(inputs[joint_idx], self.d_model)  # (batch_size, seq_len, d_model)
            # [(batch_size, seq_len, d_model)]
            joint_rep = self.embeddings[joint_idx](inputs[joint_idx])
            embed += [joint_rep]
        x = torch.stack(embed) # (n,b,seq_len,d_model)
        x = x.permute(1,2,0,3)
        # print(x)
        # (b,seq,num_joints,d)
        # x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.num_joints, self.d_model])
        # x = torch.reshape(x, (x.shape[0],x.shape[1],self.num_joints, self.d_model))
        # add the positional encoding
        inp_seq_len = inputs.shape[2]
        if self.abs_pos_encoding:
            x += self.pos_encoding[:, :inp_seq_len]
        
        # with tf.variable_scope("input_dropout", reuse=self.reuse):
        #     x = tf.layers.dropout(x, training=self.is_training, rate=dropout_rate)
        x = self.input_dropout(x)
        # print(x)

        # put into several attention layers
        # (batch_size, seq_len, num_joints, d_model)
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}
        
        look_ahead_mask = self.look_ahead_mask[:inp_seq_len, :inp_seq_len]
        for i in range(self.num_layers):
            x, block1, block2 = self.para_transformer_layers[i](x, look_ahead_mask)
            attention_weights_temporal += [block1]  # (batch_size, num_joints, num_heads, seq_len)
            attention_weights_spatial += [block2]  # (batch_size, num_heads, num_joints, num_joints)
        # (batch_size, seq_len, num_joints, d_model)
        # print(x)
        
        attention_weights['temporal'] = torch.stack(attention_weights_temporal, axis=1)  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        attention_weights['spatial'] = torch.stack(attention_weights_spatial, axis=1)  # (batch_size, num_layers, num_heads, num_joints, num_joints)

        # decode each feature to the rotation matrix space
        # different joints have different decoding matrices
        if not self.use_6d_outputs:
            # (num_joints, batch_size, seq_len, joint_size)
            # x = tf.transpose(x, [2, 0, 1, 3])
            x = x.permute(2,0,1,3)
            output = []
            for joint_idx in range(self.num_joints):
                # with tf.variable_scope("final_output_" + str(joint_idx), reuse=self.reuse):
                #     joint_output = tf.layers.dense(x[joint_idx], self.JOINT_SIZE)
                #     output += [joint_output]
                joint_output = self.output_linears[joint_idx](x[joint_idx])
                output += [joint_output]

            final_output = torch.cat(output, dim=-1)
            final_output = torch.reshape(final_output, [final_output.shape[0],
                                                        final_output.shape[1],
                                                        self.num_joints,
                                                        self.joint_size])
        else:
            # (num_joints, batch_size, seq_len, joint_size)
            # x = tf.transpose(x, [2, 0, 1, 3])
            x = x.permute(2,0,1,3)
            output = []
            for joint_idx in range(self.num_joints):
                # with tf.variable_scope("final_output_" + str(joint_idx), reuse=self.reuse):
                #     joint_output = tf.layers.dense(x[joint_idx], 6)
                #     output += [joint_output]
                joint_output = self.output_linears[joint_idx](x[joint_idx])
                output += [joint_output]
            
            n_joints = x.shape[0]
            batch_size = x.shape[1]
            seq_len = x.shape[2]

            orto6d = torch.cat(output, dim=-1)
            orto6d = torch.reshape(orto6d, [-1, 6])
            rot_mat = compute_rotation_matrix_from_ortho6d(orto6d)
            final_output = torch.reshape(rot_mat, [batch_size, seq_len, n_joints, 9])
        # print(final_output.shape) # (b,seq,n,joint_size)
        # print(inputs.shape) # (n,b,seq,joint_size)
        if self.residual_velocity:
            final_output += inputs.permute(1,2,0,3)
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
        # print(output)
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
    def sample(self,inputs,prediction_steps):
        # inputs: (b,seq_len,num_joints,joint_size)
        # prediction_steps: int, 预测步数
        # return: (b,prediction_steps,num_joints,joint_size), attentions用于分析注意力权重
        self.eval()
        predictions = []
        attentions = []
        seed_sequence = inputs
        input_sequence = seed_sequence[:,-self.window_len:,...]
        for step in range(prediction_steps):
            model_outputs, attention = self.forward(input_sequence)

            prediction = model_outputs[:,-1,...] # (b,num_joints,joint_size)
            prediction = prediction.unsqueeze(1) # (b,1,num_joints,joint_size)
            predictions += [prediction]
            attentions += [attention]

            input_sequence = torch.cat([input_sequence, prediction], dim=1)
            input_sequence = input_sequence[:, -self.window_len:]
        return torch.cat(predictions,dim=1), attentions


def train_model(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # 假设 batch 是一个张量 (B, T, J, C)
            batch = batch.to(device)

            optimizer.zero_grad()
            loss = model.get_loss(batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


