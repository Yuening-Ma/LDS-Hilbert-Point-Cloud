import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# 设置数据目录和分析结果目录
M = 'flexible'
DATA_DIR = f'datasets/ModelNet40_2/'
PLY_DIR = f"{DATA_DIR}ply_alpha/"
SAMPLE_OUTPUT_DIR = f'dim_3/8_modelnet_2/sample_M{M}/'  # 采样结果目录
SMOOTH_DIR = f'dim_3/8_modelnet_2/smooth_M{M}/'  # 平滑网格结果目录
OUTPUT_DIR = 'code_for_essay/output/exp4/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sample_ratio = 0.20

# 更新模型名称列表，使用已有的平滑网格文件模型
model_names = [
    "bottle", "chair", "cone", "cup", "flower_pot", "piano"
]

# 定义采样方法 - 更新采样方法列表，确保与smooth mesh文件匹配
sample_methods = ['srs_7', 'fps_7', 'voxel', 'bds-pca', 'bds-hilbert']

method_name_mapping = {
    'srs_7': 'SRS',
    'fps_7': 'FPS',
    'voxel': 'Voxel',
    'bds-pca': 'LDS-PCA',
    'bds-hilbert': 'LDS-Hilbert'
}

# 创建一个大图形，每行一个模型，每列一种方法
fig = plt.figure(figsize=(14, 13))  # 增大整体图形尺寸
plt.subplots_adjust(wspace=0.05, hspace=0.2)  # 减少子图之间的水平空间

# 总行数为模型数量，总列数为1（原始mesh）+ 采样方法数量
total_rows = len(model_names)
total_cols = 1 + len(sample_methods)

# 遍历每个模型
for row_idx, model_name in enumerate(model_names):
    print(f"处理模型: {model_name}")
    
    # 加载真值网格
    mesh_path = PLY_DIR + f"{model_name}_0001.ply"
    if not os.path.exists(mesh_path):
        print(f"  未找到原始网格文件: {mesh_path}")
        continue
        
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_points = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)

    # 计算中心点和半径，以便所有子图都使用相同的显示范围
    center = np.mean(mesh_points, axis=0)
    max_radius = np.max(np.linalg.norm(mesh_points - center, axis=1))
    margin = max_radius * (-0.4)  # 与exp3保持一致，使用负边距来放大显示

    # 创建第一列子图（真值网格）
    ax = fig.add_subplot(total_rows, total_cols, row_idx * total_cols + 1, projection='3d')
    ax.plot_trisurf(
        mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
        triangles=mesh_triangles, color='blue', alpha=0.3,
        edgecolor='black', linewidth=0.05
    )
    
    # 设置轴的限制，使几何体居中且放大显示
    ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
    ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
    ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
    
    # 只在第一行设置标题
    if row_idx == 0:
        ax.set_title("Ground Truth", fontsize=12, pad=0)
    
    # 在最左列显示模型名称
    ax.text2D(-0.1, 0.5, model_name, transform=ax.transAxes, fontsize=12, 
              verticalalignment='center', horizontalalignment='right')
    
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')  # 关闭坐标轴

    # 遍历每种采样方法
    for col_idx, method in enumerate(sample_methods, start=1):
        # 构建平滑网格的文件路径
        smooth_mesh_path = f"{SMOOTH_DIR}{int(sample_ratio*100)}/{model_name}_0001_{method}_smooth_mesh.ply"
        
        # 创建当前方法的子图
        ax = fig.add_subplot(total_rows, total_cols, row_idx * total_cols + col_idx + 1, projection='3d')
        
        # 检查文件是否存在
        if not os.path.exists(smooth_mesh_path):
            print(f"  未找到平滑网格文件: {smooth_mesh_path}")
            # 如果文件不存在，显示错误信息
            ax.text(0.5, 0.5, 0.5, "No mesh available", ha='center', va='center', fontsize=10)
            # 只在第一行设置标题
            if row_idx == 0:
                ax.set_title(f"{method_name_mapping[method]}", fontsize=12, pad=0)
            ax.axis('off')
            continue
        
        # 加载平滑网格
        smooth_mesh = o3d.io.read_triangle_mesh(smooth_mesh_path)
        
        # 检查网格是否为空
        if len(smooth_mesh.vertices) == 0 or len(smooth_mesh.triangles) == 0:
            print(f"  平滑网格为空: {smooth_mesh_path}")
            ax.text(0.5, 0.5, 0.5, "Empty mesh", ha='center', va='center', fontsize=10)
            # 只在第一行设置标题
            if row_idx == 0:
                ax.set_title(f"{method_name_mapping[method]}", fontsize=12, pad=0)
            ax.axis('off')
            continue
        
        # 提取顶点和三角形
        smooth_points = np.asarray(smooth_mesh.vertices)
        smooth_triangles = np.asarray(smooth_mesh.triangles)
        
        # 计算三角面数量
        num_triangles = len(smooth_triangles)
        
        # 绘制平滑网格
        ax.plot_trisurf(
            smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2],
            triangles=smooth_triangles, color='blue', alpha=0.3,
            edgecolor='black', linewidth=0.3
        )
        
        # 使用相同的中心点和半径，确保与原始模型对齐
        ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
        ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
        ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
        
        # 只在第一行设置标题
        if row_idx == 0:
            ax.set_title(f"{method_name_mapping[method]}", fontsize=12, pad=0)
        
        # 在图的右下角显示三角形数量
        ax.text2D(0.95, 0.05, f"{num_triangles}", transform=ax.transAxes, fontsize=10,
                  horizontalalignment='right', verticalalignment='bottom')
        
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')  # 关闭坐标轴

# 整体布局调整
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.1)  # 减少子图之间的空间

# 添加整体标题
# plt.suptitle("ModelNet40 Smooth Mesh Visualization", fontsize=16, y=0.98)

# 保存整个大图
output_file = f'{OUTPUT_DIR}exp4_selected_models_smooth_mesh.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"保存整体图像到: {output_file}")

print("所有模型处理完成！")