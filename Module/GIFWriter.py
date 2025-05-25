
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def draw_gif_t(verts, verts_rec, joint_to_parent, filename='compare.gif'):
    save_dir = "/home/vipuser/DL/Dataset50G/save"
    # verts:(T, N, 3)
    # verts_rec: {t:(Nt,3)} (T,N,3)
    fig = plt.figure(figsize=(10, 5))  # 设置画布大小
    # 创建两个子图
    ax1 = fig.add_subplot(121, projection='3d')  # 左边子图
    ax2 = fig.add_subplot(122, projection='3d')  # 右边子图
    interval = 1000 / 120  # 每帧的时间间隔（毫秒）
    elev = 30  # 仰角
    # 计算坐标范围
    xm = verts[:, :, 0].min()
    xM = verts[:, :, 0].max()
    ym = verts[:, :, 1].min()
    yM = verts[:, :, 1].max()
    zm = verts[:, :, 2].min()
    zM = verts[:, :, 2].max()
    disp_m = min([xm, ym, zm])
    disp_M = max([xM, yM, zM])
    def animate(i):
        # 清空子图
        ax1.clear()
        ax2.clear()
        # 绘制 verts
        # ax1.scatter(verts[i, :, 0], verts
        # [i, :, 2], verts[i, :, 1], c='b', label='Original')
        for k,v in joint_to_parent.items():
            if v is None:
                continue
            kv = [k,v]
            ax1.plot(verts[i, kv, 0], verts[i, kv, 1], verts[i, kv, 2], c='b', label='real')
        ax1.set_xlim3d([disp_m, disp_M])
        ax1.set_ylim3d([disp_m, disp_M])
        ax1.set_zlim3d([disp_m, disp_M])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('Y')
        ax1.view_init(elev=elev, azim=60)
        ax1.set_title('Original Points')
        # 绘制 verts_rec
        # ax2.scatter(verts_rec[i,:, 0], verts_rec[i,:, 2], verts_rec[i,:, 1], c='r', label='Reconstructed')
        for k,v in joint_to_parent.items():
            if v is None:
                continue
            kv = [k,v]
            ax2.plot(verts_rec[i, kv, 0], verts_rec[i, kv, 1], verts_rec[i, kv, 2], c='b', label='pred')
        ax2.set_xlim3d([disp_m, disp_M])
        ax2.set_ylim3d([disp_m, disp_M])
        ax2.set_zlim3d([disp_m, disp_M])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('Y')
        ax2.view_init(elev=elev, azim=60)
        ax2.set_title('Reconstructed Points')
        return ax1, ax2
    # 创建动画
    ani = FuncAnimation(fig, animate, frames=verts.shape[0], interval=interval)
    # 保存为 GIF
    ani.save(os.path.join(save_dir,filename), writer='pillow')
    # 关闭画布
    plt.close()
