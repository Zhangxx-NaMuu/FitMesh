import os
import time

import torch
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import open3d as o3d

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


def if_cpu():
    global device
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")


def plot_pointcloud(mesh, imgs, title=""):
    # 从网格表面均匀地采样点.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.savefig('temp.png')
    imgs.append(imageio.imread('temp.png'))
    plt.show()


def load_mesh():
    global trg_mesh, src_mesh
    # 加载网格.
    trg_obj = os.path.join('tooth.obj')
    # 我们使用load_obj读取目标3D模型
    verts, faces, aux = load_obj(trg_obj)
    """
    verts是一个形状(V, 3)的浮点数，其中V是网格中顶点的数量
    faces是一个包含以下LongTensors的对象:verts_idx, normals_idx和textures_idx
    在本教程中，法线和纹理被忽略
    """
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    """
    我们缩放归一化并将目标网格居中，以适应半径为1的圆心为(0,0,0)的球体。
    (scale, center)将用于将预测的网格带到它原来的中心和规模
    注意，标准化的目标网格，加快优化，但不是必要的
    """
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    # 我们为目标网格构造一个网格结构
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
    src_mesh = ico_sphere(4, device)
    return trg_mesh, src_mesh, scale, center
    # return trg_mesh


def gene_obj(new_src_mesh, imgs, scale, center):
    # 获取最终预测网格的顶点和面
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # 比例归一化回到原来的目标尺寸
    final_verts = final_verts * scale + center
    vis = o3d.visualization.Visualizer()
    # 使用save_obj存储预测的网格
    final_obj = os.path.join('./', 'final_model.obj')
    save_obj(final_obj, final_verts, final_faces)

    textured_mesh = o3d.io.read_triangle_mesh(final_obj)
    # print(textured_mesh)
    textured_mesh.compute_vertex_normals()

    vis.create_window(visible=False)
    vis.add_geometry(textured_mesh)
    # o3d.visualization.draw_geometries([textured_mesh], window_name="Open3D1")
    # vis.capture_screen_image("./imag.png", do_render=False)
    # time.sleep(5)
    image = vis.capture_screen_float_buffer(True)

    # vis.capture_screen_image('./image2.png', do_render=False)
    # print(1)
    image_array = np.asarray(image)
    print(image)
    plt.imsave('./image.png', image_array)
    # plt.savefig('img.png')
    # print("1")
    imgs.append(imageio.imread('./image.png'))
    # vis.close()
    # plt.show()
    # print(1)


def train():
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
    # Number of optimization steps
    Niter = 10000
    # Weight for the chamfer loss
    w_chamfer = 1.0
    # Weight for mesh edge loss
    w_edge = 1.0
    # Weight for mesh normal consistency
    w_normal = 1
    # Weight for mesh laplacian smoothing
    w_laplacian = 1
    # Plot period for the losses
    plot_period = 500
    loop = tqdm(range(Niter))
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    imgs = []
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)

        # We sample 5k points from the surface of each mesh
        sample_trg = sample_points_from_meshes(trg_mesh, 6000)
        sample_src = sample_points_from_meshes(new_src_mesh, 6000)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)

        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

        # Print the losses
        loop.set_description('total_loss = %.6f' % loss)

        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))

        # Plot mesh
        if i % plot_period == 0:
            # plot_pointcloud(new_src_mesh, imgs, title="iter: %d" % i)
            gene_obj(new_src_mesh, imgs, scale, center)
            # print(i)

        # Optimization step
        loss.backward()
        optimizer.step()
        # gene_obj(new_src_mesh, scale, center)
    imageio.mimsave('pic.gif', imgs, duration=1)


"""
我们将学习通过偏移它的顶点来变形源网格
变形参数的形状等于src_mesh中顶点的总数
"""
if __name__ == "__main__":
    if_cpu()
    imgs = []
    trg_mesh, src_mesh, scale, center = load_mesh()
    # print(trg_mesh)
    gene_obj(trg_mesh, imgs, scale, center)
    # print(imgs)
    # train()
    # mesh = load_mesh()
    # plot_mesh(mesh)
