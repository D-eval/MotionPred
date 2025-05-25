import numpy as np
from .ZXYRot import get_zxy_angle

'''
times = 20
joints_num = 25

joint_to_parent = {
    0:None,
    1:0,
    2:20,
    3:2,
    4:20,
    5:4,
    6:5,
    7:6,
    8:20,
    9:8,
    10:9,
    11:10,
    12:0,
    13:12,
    14:13,
    15:14,
    16:0,
    17:16,
    18:17,
    19:18,
    20:1,
    21:7,
    22:7,
    23:11,
    24:11
}
'''

class BVHWriter:
    def __init__(self,joint_to_parent):
        self.joint_to_parent = joint_to_parent
        joints_num = len(joint_to_parent)
        self.joints_num = joints_num
        joint_to_childs = {i:[] for i in range(joints_num)}
        for i in range(joints_num):
            if joint_to_parent[i] is not None:
                joint_to_childs[joint_to_parent[i]].append(i)
        self.joint_to_childs = joint_to_childs
        # 先从地面引一根Root
        all_joints = set(joint_to_parent.keys())
        all_parents = set(joint_to_parent.values())
        end_site = all_joints - all_parents
        self.end_site = end_site

    def cal_offset_and_motion_for_write_bvh(self,xyz):
        # xyz: (times, joints_num, 3)
        # 先计算子骨骼相对于上一个骨骼的offset,末端joint作为 End Site
        joints_num = self.joints_num
        joint_to_parent = self.joint_to_parent
        offset = np.zeros((joints_num,3))
        times = xyz.shape[0]
        for i in range(joints_num):
            if joint_to_parent[i] is not None:
                offset[i,:] = xyz[0,i,:] - xyz[0,joint_to_parent[i],:]
        all_zxy_rotation = np.zeros((times,joints_num,3))
        root_motion = np.zeros((times,3))
        for t in range(times):
            for i in range(joints_num):
                if joint_to_parent[i] is None:
                    root_motion[t] = xyz[t,i]
                    continue
                all_zxy_rotation[t,i] = get_zxy_angle(offset[i],(xyz[t,i]-xyz[t,joint_to_parent[i]]))
        return offset, all_zxy_rotation, root_motion

    def cal_offset(self,xyz):
        # xyz: (N,3)
        joints_num = self.joints_num
        joint_to_parent = self.joint_to_parent
        offset = np.zeros((joints_num,3))
        for i in range(joints_num):
            if joint_to_parent[i] is not None:
                offset[i,:] = xyz[i,:] - xyz[joint_to_parent[i],:]
        return offset

    def generate_random_xyz(self,times=10):
        joints_num = self.joints_num
        joint_to_childs = self.joint_to_childs
        joint_to_parent = self.joint_to_parent
        all_length = np.random.rand(joints_num) * 100
        local_xyz = np.zeros((times,joints_num,3))
        for t in range(times):
            for i in range(joints_num):
                temp = np.random.randn(3)
                local_xyz[t,i] = temp / np.linalg.norm(temp) * all_length[i]
        xyz = np.zeros((times,joints_num,3))
        is_certained = {i:False for i in range(joints_num)}
        is_certained[0] = True
        to_traverse = joint_to_childs[0].copy()
        while len(to_traverse)>0:
            temp_joint = to_traverse.pop(-1)
            is_certained[temp_joint] = True
            xyz[:,temp_joint,:] = xyz[:,joint_to_parent[temp_joint],:] + local_xyz[:,temp_joint,:]
            to_traverse += joint_to_childs[temp_joint]
        return xyz

    def write_bvh_from_offsets_rotations(self,offset, all_zxy_rotation, root_motion, save_path="output2.bvh"):
        '''
        offset: (num_joints, 3) 编辑模式骨骼形状
        all_zxy_rotation: (t,num_joints,3) 轴角制姿态
        root_motion: (t,3) 根运动
        '''
        joint_to_childs = self.joint_to_childs
        joint_names = [f"Joint{i}" for i in range(25)]
        frame_time = 1 / 30
        num_frames = root_motion.shape[0]
        def write_joint(f, joint_idx, indent):
            tab = "  " * indent
            name = f"Joint{joint_idx}"
            f.write(f"{tab}JOINT {name}\n")
            f.write(f"{tab}{{\n")
            off = offset[joint_idx]
            f.write(f"{tab}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}\n")
            f.write(f"{tab}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
            for child in joint_to_childs[joint_idx]:
                write_joint(f, child, indent + 1)
            if len(joint_to_childs[joint_idx]) == 0:
                f.write(f"{tab}  End Site\n")
                f.write(f"{tab}  {{\n")
                f.write(f"{tab}    OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}\n")
                f.write(f"{tab}  }}\n")
            f.write(f"{tab}}}\n")
        with open(save_path, "w") as f:
            f.write("HIERARCHY\n")
            f.write("ROOT Joint0\n")
            f.write("{\n")
            f.write("  OFFSET 0.0000 0.0000 1.0000\n") # Root可视
            f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            for child in joint_to_childs[0]:
                write_joint(f, child, 1)
            f.write("}\n")
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Frame Time: {frame_time:.4f}\n")
            for fidx in range(num_frames):
                line = ""
                root = root_motion[fidx]
                rot0 = all_zxy_rotation[fidx, 0]
                line += f"{root[0]:.4f} {root[1]:.4f} {root[2]:.4f} {rot0[0]:.4f} {rot0[1]:.4f} {rot0[2]:.4f} "
                def write_motion(joint_idx):
                    if joint_idx != 0:
                        rot = all_zxy_rotation[fidx, joint_idx]
                        line_parts.append(f"{rot[0]:.4f} {rot[1]:.4f} {rot[2]:.4f}")
                    for child in joint_to_childs[joint_idx]:
                        write_motion(child)
                line_parts = []
                for child in joint_to_childs[0]:
                    write_motion(child)
                line += " ".join(line_parts)
                f.write(line + "\n")
        return save_path


# xyz = generate_random_xyz()
# offset, all_zxy_rotation, root_motion = cal_offset_and_motion_for_write_bvh(xyz)
# write_bvh_from_offsets_rotations(offset, all_zxy_rotation, root_motion, joint_to_childs=joint_to_childs)
