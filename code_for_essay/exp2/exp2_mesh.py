import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import open3d as o3d

# 设置数据目录
M = 'int'
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'  # 原始数据目录
OUTPUT_DIR = f'code_for_essay/output/exp3/'

# 定义几何体类型
geometries = [
    "Sphere", "Cylinder", "EquilateralTriangularPrism", "Cube",
    "QuadrangularPrism", "PentagonalPrism", "HexagonalPrism", "LBeam",
    "TBeam", "UBeam", "HBeam", "Cone", "Torus", "Tetrahedron", "Octahedron", 
    # "Mobius"
]

# 创建一个 5 行 3 列的子图
fig = plt.figure(figsize=(14, 14))
axes = [fig.add_subplot(4, 4, i + 1, projection='3d') for i in range(16)]

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 绘制每个几何体的网格和原始点云
for i, geometry in enumerate(geometries):
    # 获取对应的数据 ID
    data_id = data_info[(data_info['Geometry'] == geometry) & (data_info['SampleNum'] == 9180)]['ID'].values[0]

    # 加载网格
    mesh_path = f"{DATA_DIR}{data_id:04}.ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # # 加载带噪声的点云
    # pc_noised = np.loadtxt(f"{DATA_DIR}{data_id:04}_noised.csv", delimiter=',')

    # # 绘制点云
    # axes[i].scatter(pc_noised[:, 0], pc_noised[:, 1], pc_noised[:, 2], s=3, color='red', alpha=1.0, zorder=0)


    # 绘制网格
    mesh_points = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)
    axes[i].plot_trisurf(
        mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
        triangles=mesh_triangles, color='blue', alpha=0.3, zorder=1, 
        edgecolor='black', linewidth=0.3
    )


    # 设置标题
    axes[i].set_title(geometry, fontsize=12)
    axes[i].set_xlabel('X', labelpad=5)
    axes[i].set_ylabel('Y', labelpad=5)
    axes[i].set_zlabel('Z', labelpad=5)

    axes[i].view_init(elev=30, azim=45)  # 设置统一的视角

# 最后一个axes留空
axes[-1].axis('off')
    
# 调整子图间距
plt.tight_layout(pad=1.0)  # 将四边距设置为 1.0
plt.subplots_adjust(hspace=0.2, wspace=0.2, right=0.98, left=0.07)

# 保存和显示图像
plt.savefig(f'{OUTPUT_DIR}exp3_dim3_pc_mesh.svg', format='svg')
plt.savefig(f'{OUTPUT_DIR}exp3_dim3_pc_mesh.png', dpi=300)
plt.show()