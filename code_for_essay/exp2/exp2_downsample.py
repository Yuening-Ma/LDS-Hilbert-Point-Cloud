import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

# 设置数据目录和分析结果目录
M = 'int'
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'  # 原始数据目录
SAMPLE_OUTPUT_DIR = f'dim_3/5_closed_geometry_2/sample_M{M}/'  # 降采样结果目录
ANALYZE1_DIR = f'dim_3/5_closed_geometry_2/analyze1_M{M}/'  # 分析结果目录
SMOOTH_DIR = f'dim_3/5_closed_geometry_2/smooth_M{M}/'
OUTPUT_DIR = f'code_for_essay/output/exp3/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样比例
sample_ratio = 10  # 对应子文件夹名称为 "10"

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 定义要绘制的几何体和对应的N值
plot_configs = {
    9180: ['Sphere', 'EquilateralTriangularPrism', 'Cone'],
    95760: ['LBeam', 'UBeam', 'HBeam']
}

# 定义采样方法
sample_methods = ['srs_7', 'fps_7', 'voxel', 'bds-pca', 'bds-hilbert']

method_name_mapping = {
    'srs_7': 'SRS',
    'fps_7': 'FPS',
    'voxel': 'Voxel',
    'bds-pca': 'LDS-PCA',
    'bds-hilbert': 'LDS-Hilbert'
}

# 计算总行数 - 所有要绘制的几何体数量
total_rows = sum(len(geometries) for geometries in plot_configs.values())

# 创建一个大图
fig = plt.figure(figsize=(14, 2.5*total_rows))
plt.subplots_adjust(wspace=0.3, hspace=0.1)  # 调整子图间距

# 当前行索引
current_row = 0

# 用于拆分图表的数据存储
plot_data = {}
for N in plot_configs.keys():
    plot_data[N] = []

# 遍历每个N值及其对应的几何体
for N, geometries in plot_configs.items():
    print(f"Processing N = {N}")
    
    # 遍历每个几何体
    for geometry in geometries:
        print(f"  Processing {geometry}")
        
        # 从data_info中找到对应N值和几何体类型的数据ID
        filtered_data = data_info[(data_info['SampleNum'] == N) & (data_info['Geometry'] == geometry)]
        
        if filtered_data.empty:
            print(f"No data found for {geometry} with N={N}")
            continue
        
        # 获取数据ID
        data_id = filtered_data.iloc[0]['ID']
        print(f"    Using data ID: {data_id}")
        
        # 加载真值网格
        mesh_path = f"{DATA_DIR}{data_id:04}.ply"
        if not os.path.exists(mesh_path):
            print(f"    Mesh file not found: {mesh_path}")
            continue
            
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh_points = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)
        
        # 保存数据用于拆分图表
        geometry_data = {
            'data_id': data_id,
            'geometry': geometry,
            'ground_truth_mesh': mesh,
            'smooth_meshes': {}
        }
        
        # 绘制真值网格
        ax = fig.add_subplot(total_rows, 6, current_row*6 + 1, projection='3d')
        
        # 计算三角面数量
        num_triangles = len(mesh_triangles)
        
        # 绘制真值网格
        ax.plot_trisurf(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
                       triangles=mesh_triangles, color='blue', alpha=0.3, 
                       edgecolor='black', linewidth=0.3)
        
        # 计算中心点和半径，以便放大显示
        center = np.mean(mesh_points, axis=0)
        max_radius = np.max(np.linalg.norm(mesh_points - center, axis=1))
        
        # 设置轴的限制，使几何体居中且放大显示
        margin = max_radius * (-0.4)  # 添加负边距以放大显示
        ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
        ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
        ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
        
        # 设置标题
        if current_row == 0:
            ax.set_title("Ground Truth", fontsize=12)
        
        # 设置比例和关闭坐标轴
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        
        # 在子图边缘添加几何体和N值标注
        ax.text2D(-0.1, 0.5, f"{geometry} (N={N})", transform=ax.transAxes, 
                 fontsize=10, rotation=90, va='center', ha='center')
        
        # 绘制每种采样方法的平滑网格
        for method_idx, method in enumerate(sample_methods):
            # 对应方法的平滑网格文件路径
            smooth_mesh_path = f"{SMOOTH_DIR}{sample_ratio}/{data_id:04}_{method}_smooth_mesh.ply"
            
            # 检查文件是否存在
            if not os.path.exists(smooth_mesh_path):
                print(f"    Smooth mesh file not found: {smooth_mesh_path}")
                ax = fig.add_subplot(total_rows, 6, current_row*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "No mesh available", ha='center', va='center', fontsize=10)
                if current_row == 0:
                    ax.set_title(f"{method}", fontsize=12)
                ax.axis('off')
                continue
                
            # 加载平滑网格
            smooth_mesh = o3d.io.read_triangle_mesh(smooth_mesh_path)
            
            # 存储平滑网格用于拆分图表
            geometry_data['smooth_meshes'][method] = smooth_mesh
            
            # 检查网格是否为空
            if len(smooth_mesh.vertices) == 0 or len(smooth_mesh.triangles) == 0:
                print(f"    Empty smooth mesh: {smooth_mesh_path}")
                ax = fig.add_subplot(total_rows, 6, current_row*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "Empty mesh", ha='center', va='center', fontsize=10)
                if current_row == 0:
                    ax.set_title(f"{method}", fontsize=12)
                ax.axis('off')
                continue
                
            smooth_points = np.asarray(smooth_mesh.vertices)
            smooth_triangles = np.asarray(smooth_mesh.triangles)
            
            # 绘制平滑网格
            ax = fig.add_subplot(total_rows, 6, current_row*6 + method_idx + 2, projection='3d')
            ax.plot_trisurf(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2],
                           triangles=smooth_triangles, color='blue', alpha=0.3,
                           edgecolor='black', linewidth=0.3)
            
            # 计算三角面数量
            num_smooth_triangles = len(smooth_triangles)
            
            # 计算中心点和半径，以便放大显示
            if len(smooth_points) > 0:
                center = np.mean(smooth_points, axis=0)
                max_radius = np.max(np.linalg.norm(smooth_points - center, axis=1))
                
                # 设置轴的限制，使几何体居中且放大显示
                margin = max_radius * (-0.4)  # 添加负边距以放大显示
                ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
                ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
                ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
            
            # 设置标题（只在第一行显示）
            if current_row == 0:
                ax.set_title(f"{method}", fontsize=12)
                
            # 在右下角显示三角形数量
            ax.text2D(0.95, 0.05, f"{num_smooth_triangles}", transform=ax.transAxes, 
                     fontsize=10, horizontalalignment='right', verticalalignment='bottom')
            
            # 设置比例和关闭坐标轴
            ax.set_box_aspect([1, 1, 1])
            ax.axis('off')
        
        # 添加当前几何体的数据到对应N值的列表中
        plot_data[N].append(geometry_data)
        
        # 更新行索引
        current_row += 1

# 调整布局
plt.tight_layout()

# 保存完整图形
plt.savefig(f'{OUTPUT_DIR}exp3_dim3_downsample.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}exp3_dim3_downsample.svg', format='svg', bbox_inches='tight')
plt.close()

# 创建拆分的图表
for N, geometries_data in plot_data.items():
    if not geometries_data:
        continue
        
    # 获取该N值对应的几何体数量
    n_geometries = len(geometries_data)
    
    # 创建新的图形
    fig_split = plt.figure(figsize=(14, 2.5*n_geometries))
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    
    # 绘制该N值的几何体
    for row_idx, geom_data in enumerate(geometries_data):
        data_id = geom_data['data_id']
        geometry = geom_data['geometry']
        mesh = geom_data['ground_truth_mesh']
        
        mesh_points = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)
        
        # 绘制真值网格
        ax = fig_split.add_subplot(n_geometries, 6, row_idx*6 + 1, projection='3d')
        
        # 绘制真值网格
        ax.plot_trisurf(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
                       triangles=mesh_triangles, color='blue', alpha=0.3, 
                       edgecolor='black', linewidth=0.3)
        
        # 计算中心点和半径，以便放大显示
        center = np.mean(mesh_points, axis=0)
        max_radius = np.max(np.linalg.norm(mesh_points - center, axis=1))
        
        # 设置轴的限制，使几何体居中且放大显示
        margin = max_radius * (-0.4)  # 添加负边距以放大显示
        ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
        ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
        ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
        
        # 设置标题
        if row_idx == 0:
            ax.set_title("Ground Truth", fontsize=12)
        
        # 设置比例和关闭坐标轴
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        
        # 在子图边缘添加几何体和N值标注
        ax.text2D(-0.1, 0.5, f"{geometry} (N={N})", transform=ax.transAxes, 
                 fontsize=10, rotation=90, va='center', ha='center')
        
        # 绘制每种采样方法的平滑网格
        for method_idx, method in enumerate(sample_methods):
            # 检查是否有该方法的平滑网格
            if method not in geom_data['smooth_meshes']:
                # 创建一个空子图
                ax = fig_split.add_subplot(n_geometries, 6, row_idx*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "No mesh available", ha='center', va='center', fontsize=10)
                if row_idx == 0:
                    ax.set_title(f"{method_name_mapping[method]}", fontsize=12)
                ax.axis('off')
                continue
                
            smooth_mesh = geom_data['smooth_meshes'][method]
            
            # 检查网格是否为空
            if len(np.asarray(smooth_mesh.vertices)) == 0 or len(np.asarray(smooth_mesh.triangles)) == 0:
                # 创建一个空子图
                ax = fig_split.add_subplot(n_geometries, 6, row_idx*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "Empty mesh", ha='center', va='center', fontsize=10)
                if row_idx == 0:    
                    ax.set_title(f"{method_name_mapping[method]}", fontsize=12)
                ax.axis('off')
                continue
                
            smooth_points = np.asarray(smooth_mesh.vertices)
            smooth_triangles = np.asarray(smooth_mesh.triangles)
            
            # 绘制平滑网格
            ax = fig_split.add_subplot(n_geometries, 6, row_idx*6 + method_idx + 2, projection='3d')
            ax.plot_trisurf(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2],
                           triangles=smooth_triangles, color='blue', alpha=0.3,
                           edgecolor='black', linewidth=0.3)
            
            # 计算三角面数量
            num_smooth_triangles = len(smooth_triangles)
            
            # 计算中心点和半径，以便放大显示
            if len(smooth_points) > 0:
                center = np.mean(smooth_points, axis=0)
                max_radius = np.max(np.linalg.norm(smooth_points - center, axis=1))
                
                # 设置轴的限制，使几何体居中且放大显示
                margin = max_radius * (-0.4)  # 添加负边距以放大显示
                ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
                ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
                ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
            
            # 设置标题（只在第一行显示）
            if row_idx == 0:
                ax.set_title(f"{method_name_mapping[method]}", fontsize=12)
                
            # 在右下角显示三角形数量
            ax.text2D(0.95, 0.05, f"{num_smooth_triangles}", transform=ax.transAxes, 
                     fontsize=10, horizontalalignment='right', verticalalignment='bottom')
            
            # 设置比例和关闭坐标轴
            ax.set_box_aspect([1, 1, 1])
            ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存拆分图形
    plt.savefig(f'{OUTPUT_DIR}exp3_dim3_downsample_{N}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}exp3_dim3_downsample_{N}.svg', format='svg', bbox_inches='tight')
    plt.close()

print("Done!")
