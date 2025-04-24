import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 设置数据目录
M = 'int'
DATA_DIR = f'dim_3/4_surface_noised/data_M{M}/'  # 原始数据目录
OUTPUT_DIR = f'code_for_essay/output/exp2/'

# 定义分布类型和 N 值
surfaces = ['Plane', 'Sphere', 'Ellipsoid', 'Torus', 'Cylinder', 'Cone', 'Paraboloid']
n_value = 9180  # 选择 N=340 的数据

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')
dist_data = data_info[data_info['DataPoints'] == n_value]

# 创建一个 2 行 4 列的子图
fig = plt.figure(figsize=(14, 7))
axes = [fig.add_subplot(2, 4, i + 1, projection='3d') for i in range(8)]

# 绘制每个曲面的真值和原始点云
for i, surface in enumerate(surfaces):
    # 获取对应的数据 ID
    data_id = dist_data[dist_data['Surface'] == surface]['ID'].values[0]

    # 读取原始点云
    pc_original = np.loadtxt(f"{DATA_DIR}{data_id:04}_noised.csv", delimiter=',')

    # 获取曲面参数
    params = dist_data[dist_data['Surface'] == surface]['Parameters'].values[0].split('_')
    params = list(map(float, params))

    # 生成真值曲面
    if surface == "Plane":
        a, b, c = params
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        x, y = np.meshgrid(x, y)
        z = a * x + b * y + c
    elif surface == "Sphere":
        x_center, y_center, z_center, radius = params
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = x_center + radius * np.sin(phi) * np.cos(theta)
        y = y_center + radius * np.sin(phi) * np.sin(theta)
        z = z_center + radius * np.cos(phi)
    elif surface == "Ellipsoid":
        x_center, y_center, z_center, a, b, c = params
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = x_center + a * np.sin(phi) * np.cos(theta)
        y = y_center + b * np.sin(phi) * np.sin(theta)
        z = z_center + c * np.cos(phi)
    elif surface == "Torus":
        x_center, y_center, z_center, R, r = params
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = x_center + (R + r * np.cos(phi)) * np.cos(theta)
        y = y_center + (R + r * np.cos(phi)) * np.sin(theta)
        z = z_center + r * np.sin(phi)
    elif surface == "Cylinder":
        x_center, y_center, radius = params
        z_center = 0
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(-2, 2, 100)
        theta, z = np.meshgrid(theta, z)
        x = x_center + radius * np.cos(theta)
        y = y_center + radius * np.sin(theta)
        z = z_center + z
    elif surface == "Cone":
        x_center, y_center, z_center, radius = params
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(0, 2, 100)
        theta, z = np.meshgrid(theta, z)
        x = x_center + radius * z * np.cos(theta)
        y = y_center + radius * z * np.sin(theta)
        z = z_center + z
    elif surface == "Paraboloid":
        x_center, y_center, z_center, a = params
        r = np.linspace(0, 2, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        r, theta = np.meshgrid(r, theta)
        x = x_center + r * np.cos(theta)
        y = y_center + r * np.sin(theta)
        z = z_center + a * r**2

    # 绘制真值曲面
    axes[i].plot_surface(x, y, z, color='blue', alpha=0.3)

    # 绘制原始点云
    # axes[i].scatter(pc_original[:, 0], pc_original[:, 1], pc_original[:, 2], s=1, color='red', alpha=1.0)

    # 设置标题
    axes[i].set_title(surface, fontsize=12, pad=2)

    # 计算坐标轴范围
    margin = 0.5  # 增加的边界范围
    x_min, x_max = pc_original[:, 0].min() - margin, pc_original[:, 0].max() + margin
    y_min, y_max = pc_original[:, 1].min() - margin, pc_original[:, 1].max() + margin
    z_min, z_max = pc_original[:, 2].min() - margin, pc_original[:, 2].max() + margin

    # 设置坐标轴范围
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_zlim(z_min, z_max)

    axes[i].set_xlabel('X', fontsize=10)
    axes[i].set_ylabel('Y', fontsize=10)
    axes[i].set_zlabel('Z', fontsize=10, labelpad=4)

# 最后两个子图为空
axes[7].axis('off')

# 调整子图间距
plt.tight_layout(pad=0.5)  # 减小四边距
plt.subplots_adjust(hspace=0.15, wspace=0.15, left=0.02, right=0.96, top=0.95, bottom=0.05)  # 调整边距和间距

# 保存和显示图像
plt.savefig(f'{OUTPUT_DIR}exp2_dim3_pc_surface_N{n_value}.svg', format='svg')
plt.savefig(f'{OUTPUT_DIR}exp2_dim3_pc_surface_N{n_value}.png', dpi=300)
plt.show()