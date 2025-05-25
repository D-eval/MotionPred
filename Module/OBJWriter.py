


def write_obj(filepath, vertices, faces=None):
    """
    将 (N, 3) 顶点坐标数组写入 .obj 文件

    参数:
        filepath: str，输出的 .obj 文件路径
        vertices: numpy.ndarray，形状为 (N, 3)
        faces: numpy.ndarray，可选，形状为 (F, 3)，每行是一个三角面（索引从 0 开始）
    """
    with open(filepath, 'w') as f:
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # 写入面（可选）
        if faces is not None:
            for face in faces:
                # .obj 索引从 1 开始，所以加 1
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# write_obj('/home/vipuser/DL/Dataset50G/save/a.obj',bm.shapedirs[:,:,0])

# write_obj('/home/vipuser/DL/Dataset50G/save/pose_offsets.obj',pose_offsets[0,:,:])