import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
# from pytorch3d.utils import ico_sphere
#
# src_mesh = ico_sphere(4, device)
fig = plt.figure(figsize=(10, 10))
ax1 = Axes3D(fig)
ax1.plot3D([0, 1], [0, 1], [0, 1], 'red')
x_track = np.zeros((1, 3))
x_track_s = np.array([.0, .0, .0])
theta = 0


def gen_path():
    global x_track_s, x_track, theta
    theta += 10 * np.pi / 180
    x = 6 * np.sin(theta)
    y = 6 * np.cos(theta)
    x_track_s += [x, y, 0.1]
    x_track = np.append(x_track, [x_track_s], axis=0)
    return x_track


def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    # 更新直线和x轴（用一个新的x轴的标签）。
    # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
    x_track = gen_path()
    ax1.set_xlabel(label)

    ax1.plot3D(x_track[:, 0], x_track[:, 1], x_track[:, 2], 'blue')
    return ax1


if __name__ == '__main__':
    # FuncAnimation 会在每一帧都调用“update” 函数。
    # 在这里设置一个10帧的动画，每帧之间间隔200毫秒
    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    # if len(sys.argv) > 1 and sys.argv[1] == 'save':
    anim.save('line.gif', dpi=80, writer='imagemagick')
    # else:
    #     # plt.show() 会一直循环播放动画
    #     plt.show()
