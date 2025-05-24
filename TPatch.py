

import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points



# 输入一个点集Tensor(N,3),一个点的索引,采样个数
# 返回采样点的索引,最小距离Tensor
def FPS(verts,idx_start=0,num_sample = 30):
    # verts:(N,3)
    # idx_selects:(num_sample)
    idx_selects = torch.zeros(num_sample,dtype=torch.long)
    idx_selects[0] = idx_start

    vert_temp = verts[idx_start,:]
    vert_temp = vert_temp.unsqueeze(0)
    dist_sq = ((verts - vert_temp)**2).sum(dim=1)

    for i in range(1,num_sample):
        idx_max = dist_sq.argmax()
        idx_selects[i] = idx_max
        vert_temp = verts[idx_max,:]
        vert_temp = vert_temp.unsqueeze(0)
        dist_sq_temp = ((verts - vert_temp)**2).sum(dim=1)
        idx_change = dist_sq_temp < dist_sq
        dist_sq[idx_change] = dist_sq_temp[idx_change]
    return idx_selects, dist_sq


def furthest_point_sampling(verts, npoint):
    # verts: (B,N,3)
    # npoint: int32
    B,N,_ = verts.shape
    idx = torch.zeros(B,npoint).to(verts.device)
    for b in range(B):
        idx_temp, _ = FPS(verts[b],0,npoint)
        idx[b,:] = idx_temp
    return idx


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        fps_inds = furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = FurthestPointSampling.apply



def sample_and_group_all_4d(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, t, N, C = xyz.shape
    new_xyz = torch.zeros(B, t, 1, C).to(device)
    grouped_xyz = xyz.view(B, t, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, t, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    idx = idx.long()
    new_points = points[batch_indices, idx, :]
    return new_points



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group_4d(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, t, N, 3]
        points: input points data, [B, t, N, D] None
    Return:
        new_xyz: sampled points position data, [B, t, npoint, nsample, 3]
        new_points: sampled points data, [B, t, npoint, nsample, 3+D]
    """
    B, t, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz[:,0,...], npoint)  # [B, npoint, C]
    xyz = xyz.reshape(-1, N, C)
    # xyz = torch.flatten(xyz,0,1)
    new_xyz = index_points(xyz, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B*t, S, 1, C)  # shouldnt this also be scaled to unit sphere?

    if points is not None:
        D = points.shape[-1]
        grouped_points = index_points(points.reshape(-1, N, D), idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    new_xyz = new_xyz.reshape(B, t, npoint, C)
    new_points = new_points.reshape(B, t, npoint, nsample, -1)

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points



class PointNetPP4DSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, temporal_conv=4):
        super(PointNetPP4DSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, 1], 1, padding='same'))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel
        self.group_all = group_all # True, 骨骼点很少

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: (B,T,D,N)
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        b, t, k, n = xyz.shape
        #xyz = xyz.reshape(-1, k, n) # (B*T, 3, N)
        #xyz = xyz.permute(0, 2, 1) # (B*T, N, 3)
        if points is not None:
            points = points.permute(0, 1, 3, 2) # (B,T,D,N)
            # points = points.reshape(-1, points.shape[-2], points.shape[-1])

        # (b,t,n,k)
        if self.group_all:
            # 聚合后的点, 每个点拼接整体特征
            # (B,1,3), (B,1,N,3+D)
            new_xyz, new_points = sample_and_group_all_4d(xyz.permute(0,1,3,2), points)
        else:
            # (B,T,N1,M1,3) (B,T,N1,M1,3+D)
            new_xyz, new_points = sample_and_group_4d(self.npoint, self.radius, self.nsample,
                                                      xyz.permute(0,1,3,2), points)
        new_points = new_points.reshape(b, t, new_points.shape[-3], new_points.shape[-2], new_points.shape[-1])
        new_points = new_points.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]
        # (B,D,N,T,M)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points))) # 并无拼接
        # (B,D,N,T,M)
        new_points = torch.max(new_points, -1)[0]
        new_points = new_points.permute(0, 3, 1, 2)
        new_xyz = new_xyz.reshape(b, t, new_xyz.shape[-2], new_xyz.shape[-1]).permute(0, 1, 3, 2)
        return new_xyz, new_points



class PointNetPP4D(nn.Module):
    def __init__(self,config):
        super(PointNetPP4D, self).__init__()
        fps = config['fps']
        in_channel = config['in_channel']
        self.fps = fps # 采样率
        self.in_channel = in_channel
        temporal_conv = fps // 10
        # self.sa1 = PointNetPP4DSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel,
        #                                       mlp=[64, 64, 128], group_all=False, temporal_conv=8)
        # self.sa2 = PointNetPP4DSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
        #                                       mlp=[128, 128, 256], group_all=False, temporal_conv=4)
        self.sa = PointNetPP4DSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=in_channel,
                                              mlp=[256, 512, 1024], group_all=True, temporal_conv=temporal_conv)
        # self.temporal_pool1 = torch.nn.MaxPool2d([4, 1])
        # self.temporal_pool2 = torch.nn.MaxPool2d([4, 1])
        # self.temporal_pool_xyz = torch.nn.AvgPool2d([4, 1])
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)
        # self.temporalconv = torch.nn.Conv1d(256, 256, n_frames, 1, padding='same')
        # self.bn3 = nn.BatchNorm1d(256)
    def forward(self, xyz):
        # xyz: (b,t,d,n)
        # return: (b,D)
        b, t, d, n = xyz.shape # (b,t,d,n)
        # new_B = B*t
        # l1_xyz, l1_points = self.sa1(xyz, points=None)
        # l1_xyz = l1_xyz.permute(0,3,1,2)
        # l1_points = l1_points.permute(0,3,1,2)
        # l1_xyz, l1_points = self.temporal_pool_xyz(l1_xyz), self.temporal_pool1(l1_points)
        # l1_xyz = l1_xyz.permute(0,2,3,1)
        # l1_points = l1_points.permute(0,2,3,1)
        # 
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l2_xyz = l2_xyz.permute(0,3,1,2)
        # l2_points = l2_points.permute(0,3,1,2)
        # l2_xyz, l2_points = self.temporal_pool_xyz(l2_xyz), self.temporal_pool2(l2_points)
        # l2_xyz = l2_xyz.permute(0,2,3,1)
        # l2_points = l2_points.permute(0,2,3,1)
        
        l_xyz, l_points = self.sa(xyz, points=None)
        # # (b,T//4//4,3,N=1), (b, T//4//4, D, N=1)
        

        # l3_xyz, l3_points = l3_xyz.squeeze(-1), l3_points.squeeze(-1)
        # x = l3_points.permute(0, 2, 1)
 
        # x = F.interpolate(x, t, mode='linear', align_corners=True).permute(0, 2, 1)

        # x = x.reshape(b*t, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        # # learn a temporal filter on all per-frame global representations
        # x = F.relu(self.bn3(self.temporalconv(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        # x = self.fc3(x)
        # # x:(B,num_class,T)
        # x = F.log_softmax(x, -1)

        # return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': l3_points}
        return l_xyz.mean(1)[-1]

    # def replace_logits(self, num_classes):
    #     self._num_classes = num_classes
    #     self.fc3 = nn.Linear(256, num_classes)



''' 用法示例:
device = torch.device('cuda')

num_class = 10
n_frames=32
in_channel=3

model = PointNetPP4D(model_cfg=None,num_class=num_class,n_frames=n_frames,in_channel=in_channel).to(device)


# dataset
b = 2
t = 35
d = 3
n = 512
xyz = torch.randn((b,t,d,n)).to(device)

# forward
output = model(xyz)

prob = output['pred']
fea = output['features']
z = fea.mean(1)
'''


'''
b, t, d, n = xyz.shape # (B,T,D,N)
# new_B = B*t
l1_xyz, l1_points = model.sa1(xyz, points=None)
l1_xyz = l1_xyz.permute(0,3,1,2)
l1_points = l1_points.permute(0,3,1,2)
l1_xyz, l1_points = model.temporal_pool_xyz(l1_xyz), model.temporal_pool1(l1_points)
l1_xyz = l1_xyz.permute(0,2,3,1)
l1_points = l1_points.permute(0,2,3,1)

l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
l2_xyz = l2_xyz.permute(0,3,1,2)
l2_points = l2_points.permute(0,3,1,2)
l2_xyz, l2_points = model.temporal_pool_xyz(l2_xyz), model.temporal_pool2(l2_points)
l2_xyz = l2_xyz.permute(0,2,3,1)
l2_points = l2_points.permute(0,2,3,1)

l3_xyz, l3_points = model.sa3(l2_xyz, l2_points)
# # (b,T,3,N=1), (b,T,D,N=1)


l3_xyz, l3_points = l3_xyz.squeeze(-1), l3_points.squeeze(-1)

x = l3_points.permute(0, 2, 1)

x = F.interpolate(x, t, mode='linear', align_corners=True).permute(0, 2, 1)

x = x.reshape(b*t, 1024)
x = model.drop1(F.relu(model.bn1(model.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
x = model.drop2(F.relu(model.bn2(model.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
# learn a temporal filter on all per-frame global representations
x = F.relu(model.bn3(model.temporalconv(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
x = model.fc3(x)

x = F.log_softmax(x, -1)
'''

'''
# b, t, d, n = xyz.shape # (B,T,D,N)
# new_B = B*t
l1_xyz, l1_points = sa1(xyz, points=points)
l1_xyz = l1_xyz.permute(0,3,1,2)
l1_points = l1_points.permute(0,3,1,2)
l1_xyz, l1_points = temporal_pool_xyz(l1_xyz), temporal_pool1(l1_points)
l1_xyz = l1_xyz.permute(0,2,3,1)
l1_points = l1_points.permute(0,2,3,1)

l2_xyz, l2_points = sa2(l1_xyz, l1_points)
l2_xyz = l2_xyz.permute(0,3,1,2)
l2_points = l2_points.permute(0,3,1,2)
l2_xyz, l2_points = temporal_pool_xyz(l2_xyz), temporal_pool2(l2_points)
l2_xyz = l2_xyz.permute(0,2,3,1)
l2_points = l2_points.permute(0,2,3,1)

l3_xyz, l3_points = sa3(l2_xyz, l2_points)
# (b,N=1,3,T), (b,N=1,D,T)
'''

'''
l3_xyz, l3_points = l3_xyz.squeeze(-1), l3_points.squeeze(-1)
x = l3_points.permute(0, 2, 1)

x = F.interpolate(x, t, mode='linear', align_corners=True).permute(0, 2, 1)

x = x.reshape(b*t, 1024)
x = drop1(F.relu(bn1(fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
x = drop2(F.relu(bn2(fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
# learn a temporal filter on all per-frame global representations
x = F.relu(bn3(temporalconv(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
x = fc3(x)

x = F.log_softmax(x, -1)
'''

'''
B, t, N, C = xyz.shape
S = sa1.nsample
D = points.shape[-1]

xyz = xyz.permute(0,1,3,2)
fps_idx = farthest_point_sample(xyz[:,0,...], S)

xyz[0,0,:,:]
FPS(verts[b],0,npoint)
'''


'''
b, t, k, n = xyz.shape
#xyz = xyz.reshape(-1, k, n) # (B*T, 3, N)
#xyz = xyz.permute(0, 2, 1) # (B*T, N, 3)
if points is not None:
    points = points.permute(0, 1, 3, 2) # (B,T,D,N)
    # points = points.reshape(-1, points.shape[-2], points.shape[-1])

new_xyz, new_points = sample_and_group_4d(sa1.npoint, sa1.radius, sa1.nsample,
                                        xyz.permute(0,1,3,2), points)

npoint = sa1.npoint
radius = sa1.radius
nsample = sa1.nsample
xyz = xyz.permute(0,1,3,2)
points = points

# 
B, t, N, C = xyz.shape
S = npoint
D = points.shape[-1]
fps_idx = farthest_point_sample(xyz[:,0,...], npoint)  # [B, npoint, C]

xyz = torch.flatten(xyz,0,1)

new_xyz = index_points(xyz, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, npoint))
idx = query_ball_point(radius, nsample, xyz, new_xyz)
grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
grouped_xyz_norm = grouped_xyz - new_xyz.view(B*t, S, 1, C)  # shouldnt this also be scaled to unit sphere?

if points is not None:
    grouped_points = index_points(points.reshape(-1, N, D), idx)
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
else:
    new_points = grouped_xyz_norm

new_xyz = new_xyz.reshape(B, t, npoint, C)
new_points = new_points.reshape(B, t, npoint, nsample, -1)
'''