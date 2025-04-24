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

# 数据点数列表
n_values = [9180, 37240, 95760]

# 定义几何体类型，保持有序
geometries = [
    "Sphere", "Cylinder", 
    "EquilateralTriangularPrism", 
    "Cube",
    "QuadrangularPrism", "PentagonalPrism", "HexagonalPrism", 
    "LBeam", "TBeam", "UBeam", "HBeam",
    "Cone", "Torus", "Tetrahedron", "Octahedron"
]

# 定义采样方法
sample_methods = ['srs_7', 'fps_7', 'voxel', 'bds-pca', 'bds-hilbert']

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

num_in_group_A = 8
num_in_group_B = 7

# 对每个N值分别处理
for N in n_values:
    print(f"Processing N = {N}")
    
    # 筛选出对应N值的数据
    filtered_data = data_info[data_info['SampleNum'] == N].copy()
    
    # 按几何体类型排序（使用自定义顺序）
    filtered_data['GeometryOrder'] = filtered_data['Geometry'].apply(lambda x: geometries.index(x) if x in geometries else 999)
    filtered_data = filtered_data.sort_values('GeometryOrder')
    
    # 第一组：前10个几何体
    first_group = filtered_data.iloc[:num_in_group_A]
    # 第二组：后5个几何体
    second_group = filtered_data.iloc[num_in_group_A:num_in_group_A+num_in_group_B]
    
    # 创建第一组图形（10×6）- 调整更紧凑
    fig1 = plt.figure(figsize=(14, 16))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # 减少子图之间的空间
    
    # 创建第二组图形（5×6）- 调整更紧凑
    fig2 = plt.figure(figsize=(14, 14))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # 减少子图之间的空间
    
    # 处理第一组几何体（10个）
    for idx, (_, row) in enumerate(first_group.iterrows()):
        data_id = row['ID']
        geometry = row['Geometry']
        print(f"  Processing {geometry} (ID: {data_id})")

        # 加载真值网格
        mesh_path = f"{DATA_DIR}{data_id:04}.ply"
        if not os.path.exists(mesh_path):
            print(f"    Mesh file not found: {mesh_path}")
            continue
            
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh_points = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)

        # 创建第一组子图
        ax = fig1.add_subplot(num_in_group_A, 6, idx*6 + 1, projection='3d')
        surf = ax.plot_trisurf(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
                       triangles=mesh_triangles, color='blue', alpha=0.3, 
                       edgecolor='black', linewidth=0.3)
        
        # 计算中心点和半径，以便放大显示
        center = np.mean(mesh_points, axis=0)
        max_radius = np.max(np.linalg.norm(mesh_points - center, axis=1))
        
        # 设置轴的限制，使几何体居中且放大显示
        margin = max_radius * (-0.4)  # 添加10%的边距
        ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
        ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
        ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
        
        # 只有第一行设置标题
        if idx == 0:
            ax.set_title("Ground Truth", fontsize=12, pad=0)  # 减少标题和图之间的间距
        
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')  # 关闭坐标轴
        
        # 绘制每种采样方法的平滑网格
        for method_idx, method in enumerate(sample_methods):
            smooth_mesh_path = ""
            
            if method.startswith(('srs', 'fps')):
                # 尝试寻找对应的平滑网格文件
                smooth_mesh_path = f"{SMOOTH_DIR}{sample_ratio}/{data_id:04}_{method}_smooth_mesh.ply"

            else:
                smooth_mesh_path = f"{SMOOTH_DIR}{sample_ratio}/{data_id:04}_{method}_smooth_mesh.ply"

            
            # 检查文件是否存在
            if not os.path.exists(smooth_mesh_path):
                print(f"    Smooth mesh file not found: {smooth_mesh_path}")
                ax = fig1.add_subplot(num_in_group_A, 6, idx*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "No mesh available", ha='center', va='center', fontsize=10)
                # 只有第一行设置标题
                if idx == 0:
                    ax.set_title(f"{method}", fontsize=12, pad=0)
                ax.axis('off')
                continue
                
            # 加载平滑网格
            smooth_mesh = o3d.io.read_triangle_mesh(smooth_mesh_path)
            
            # 检查网格是否为空
            if len(smooth_mesh.vertices) == 0 or len(smooth_mesh.triangles) == 0:
                print(f"    Empty smooth mesh: {smooth_mesh_path}")
                ax = fig1.add_subplot(num_in_group_A, 6, idx*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "Empty mesh", ha='center', va='center', fontsize=10)
                # 只有第一行设置标题
                if idx == 0:
                    ax.set_title(f"{method}", fontsize=12, pad=0)
                ax.axis('off')
                continue
                
            smooth_points = np.asarray(smooth_mesh.vertices)
            smooth_triangles = np.asarray(smooth_mesh.triangles)
            
            # 绘制平滑网格
            ax = fig1.add_subplot(num_in_group_A, 6, idx*6 + method_idx + 2, projection='3d')
            ax.plot_trisurf(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2],
                           triangles=smooth_triangles, color='blue', alpha=0.3,
                           edgecolor='black', linewidth=0.3)
            
            # 计算中心点和半径，以便放大显示
            if len(smooth_points) > 0:
                center = np.mean(smooth_points, axis=0)
                max_radius = np.max(np.linalg.norm(smooth_points - center, axis=1))
                
                # 设置轴的限制，使几何体居中且放大显示
                margin = max_radius * (-0.4)  # 添加10%的边距
                ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
                ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
                ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
            
            # 只有第一行设置标题
            if idx == 0:
                ax.set_title(f"{method}", fontsize=12, pad=0)
                
            ax.set_box_aspect([1, 1, 1])
            ax.axis('off')  # 关闭坐标轴
    
    # 处理第二组几何体（后5个）
    for idx, (_, row) in enumerate(second_group.iterrows()):
        data_id = row['ID']
        geometry = row['Geometry']
        print(f"  Processing {geometry} (ID: {data_id})")

        # 加载真值网格
        mesh_path = f"{DATA_DIR}{data_id:04}.ply"
        if not os.path.exists(mesh_path):
            print(f"    Mesh file not found: {mesh_path}")
            continue
            
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh_points = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)

        # 创建第二组子图
        ax = fig2.add_subplot(num_in_group_B, 6, idx*6 + 1, projection='3d')
        ax.plot_trisurf(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
                       triangles=mesh_triangles, color='blue', alpha=0.3,
                       edgecolor='black', linewidth=0.3)
        
        # 计算中心点和半径，以便放大显示
        center = np.mean(mesh_points, axis=0)
        max_radius = np.max(np.linalg.norm(mesh_points - center, axis=1))
        
        # 设置轴的限制，使几何体居中且放大显示
        margin = max_radius * (-0.4)
        ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
        ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
        ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
        
        # 只有第一行设置标题
        if idx == 0:
            ax.set_title("Ground Truth", fontsize=12, pad=0)
            
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')  # 关闭坐标轴
        
        # 绘制每种采样方法的平滑网格
        for method_idx, method in enumerate(sample_methods):
            smooth_mesh_path = ""
            
            if method.startswith(('srs', 'fps')):
                # 尝试寻找对应的平滑网格文件
                smooth_mesh_path = f"{SMOOTH_DIR}{sample_ratio}/{data_id:04}_{method}_smooth_mesh.ply"

            else:
                smooth_mesh_path = f"{SMOOTH_DIR}{sample_ratio}/{data_id:04}_{method}_smooth_mesh.ply"

            
            # 检查文件是否存在
            if not os.path.exists(smooth_mesh_path):
                print(f"    Smooth mesh file not found: {smooth_mesh_path}")
                ax = fig2.add_subplot(num_in_group_B, 6, idx*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "No mesh available", ha='center', va='center', fontsize=10)
                # 只有第一行设置标题
                if idx == 0:
                    ax.set_title(f"{method}", fontsize=12, pad=0)
                ax.axis('off')
                continue
                
            # 加载平滑网格
            smooth_mesh = o3d.io.read_triangle_mesh(smooth_mesh_path)
            
            # 检查网格是否为空
            if len(smooth_mesh.vertices) == 0 or len(smooth_mesh.triangles) == 0:
                print(f"    Empty smooth mesh: {smooth_mesh_path}")
                ax = fig2.add_subplot(num_in_group_B, 6, idx*6 + method_idx + 2, projection='3d')
                ax.text(0.5, 0.5, 0.5, "Empty mesh", ha='center', va='center', fontsize=10)
                # 只有第一行设置标题
                if idx == 0:
                    ax.set_title(f"{method}", fontsize=12, pad=0)
                ax.axis('off')
                continue
                
            smooth_points = np.asarray(smooth_mesh.vertices)
            smooth_triangles = np.asarray(smooth_mesh.triangles)
            
            # 绘制平滑网格
            ax = fig2.add_subplot(num_in_group_B, 6, idx*6 + method_idx + 2, projection='3d')
            ax.plot_trisurf(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2],
                           triangles=smooth_triangles, color='blue', alpha=0.3,
                           edgecolor='black', linewidth=0.3)
            
            # 计算中心点和半径，以便放大显示
            if len(smooth_points) > 0:
                center = np.mean(smooth_points, axis=0)
                max_radius = np.max(np.linalg.norm(smooth_points - center, axis=1))
                
                # 设置轴的限制，使几何体居中且放大显示
                margin = max_radius * (-0.4)  # 添加10%的边距
                ax.set_xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
                ax.set_ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
                ax.set_zlim(center[2] - max_radius - margin, center[2] + max_radius + margin)
            
            # 只有第一行设置标题
            if idx == 0:
                ax.set_title(f"{method}", fontsize=12, pad=0)
                
            ax.set_box_aspect([1, 1, 1])
            ax.axis('off')  # 关闭坐标轴
    
    # 调整图形布局，更紧凑
    fig1.tight_layout(pad=1)
    fig2.tight_layout(pad=1)
    
    # 保存图形，png和svg
    fig1.savefig(f'{OUTPUT_DIR}exp3_{N}_group1_smoothed_mesh.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{OUTPUT_DIR}exp3_{N}_group2_smoothed_mesh.png', dpi=300, bbox_inches='tight')
    
    # 关闭图形以释放内存
    plt.close(fig1)
    plt.close(fig2)
    
    print(f"Saved figures for N = {N}")

print("Done!")
