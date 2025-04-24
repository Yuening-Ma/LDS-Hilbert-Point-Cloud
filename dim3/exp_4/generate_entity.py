import os
import open3d as o3d
import numpy as np
import pymeshlab
import sys
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')
from sample_evaluation import *

DATA_DIR = "dim_3/7_stanford_all/Stanford_all/"

ply_dir = f"{DATA_DIR}ply/"
ply_scaled_dir = f"{DATA_DIR}ply_scaled/"
pc_dir = f"{DATA_DIR}pc/"
pc_noise_dir = f"{DATA_DIR}pc_noise/"

os.makedirs(ply_scaled_dir, exist_ok=True)
os.makedirs(pc_dir, exist_ok=True)
os.makedirs(pc_noise_dir, exist_ok=True)

# sample_num = 90000
sigma = 0.5

def generate_pc_from_mesh(mesh, sample_method="poisson", pc_num=10000, sigma=None):
    '''
    从mesh生成点云
    '''
    if sample_method == 'uniform':
        pcd = mesh.sample_points_uniformly(number_of_points=pc_num)
    else:
        pcd = mesh.sample_points_poisson_disk(number_of_points=pc_num)

    points = np.asarray(pcd.points)
    
    if sigma is None:
        return points
    else:
        noise = np.random.normal(0, sigma, size=points.shape)
        points += noise
        return points

# 处理所有OFF文件
ply_names = os.listdir(ply_dir)
ply_names.sort()

for i, ply_name in enumerate(ply_names):
    if not ply_name.endswith('.ply'):
        continue
        
    model_name = ply_name.split('.')[0]
    print(f"Processing {model_name} ({i+1}/{len(ply_names)})")
    
    # 源文件和目标文件路径
    ply_path = os.path.join(ply_dir, ply_name)
    ply_scaled_path = os.path.join(ply_scaled_dir, ply_name)
    pc_path = os.path.join(pc_dir, f"{model_name}.csv")
    pc_noise_path = os.path.join(pc_noise_dir, f"{model_name}.csv")

    try:
        # 1. 加载点云并进行缩放
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # 使用Open3D进行缩放，使高度标准化为100
        bbox = pcd.get_axis_aligned_bounding_box()
        height = bbox.get_max_bound()[2] - bbox.get_min_bound()[2]
        if height > 0:
            scale_factor = 100.0 / height
            pcd_scaled = pcd.scale(scale_factor, center=pcd.get_center())
            # print(f"Scaled point cloud to height 100, scale factor: {scale_factor}")
        
        # 打印点云点数
        print(f"Point number: {len(pcd_scaled.points)}")
        # 可视化
        # o3d.visualization.draw_geometries([pcd_scaled])

        # 2. 获取点云的点作为pc
        pc = np.asarray(pcd_scaled.points)
        
        # 3. 在pc基础上增加高斯噪声
        pc_noise = pc + np.random.normal(0, sigma, size=pc.shape)
        # 创建pcd_noise，并可视化
        pcd_noise = o3d.geometry.PointCloud()
        pcd_noise.points = o3d.utility.Vector3dVector(pc_noise)
        # o3d.visualization.draw_geometries([pcd_noise])
        
        # 4. 保存点云
        np.savetxt(pc_path, pc, delimiter=',', fmt="%.6f")
        np.savetxt(pc_noise_path, pc_noise, delimiter=',', fmt="%.6f")
        
        # 5. 使用get_smooth_mesh函数进行三维重建
        d = round((27000 / len(pcd_scaled.points)) ** 0.5, 1)
        print(f"d: {d}")
        radii = [0.55 * d, 0.75 * d, 1 * d, 2 * d, 3 * d]
        mesh_reconstructed, smoothed_mesh_with_isolated = get_smooth_mesh(pcd_scaled, radius=3, k=30, radii=radii)

        # 打印三角面数
        print(f"Triangle number: {len(mesh_reconstructed.triangles)}, Vertex number: {len(mesh_reconstructed.vertices)}")
        # o3d.visualization.draw_geometries([mesh_reconstructed], mesh_show_wireframe=True)
        
        # 6. 保存重建的网格
        if mesh_reconstructed is not None:
            o3d.io.write_triangle_mesh(ply_scaled_path, mesh_reconstructed)
            print(f"Saved reconstructed mesh for {model_name}")
        else:
            print(f"Failed to reconstruct mesh for {model_name}")
        
        print(f"Saved point clouds for {model_name}")
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

print("Done processing all models")