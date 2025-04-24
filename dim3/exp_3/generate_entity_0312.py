import os
import open3d as o3d
import numpy as np
import pymeshlab

DATA_DIR = "datasets/ModelNet40_2/"

off_dir = f"{DATA_DIR}off/"
ply_alpha_dir = f"{DATA_DIR}ply_alpha/"
pc_dir = f"{DATA_DIR}pc/"
pc_noise_dir = f"{DATA_DIR}pc_noise/"

os.makedirs(ply_alpha_dir, exist_ok=True)
os.makedirs(pc_dir, exist_ok=True)
os.makedirs(pc_noise_dir, exist_ok=True)

sample_num = 90000
sigma = 1

def generate_alpha_ply_from_off(off_path, ply_path, alpha=0.05, offset=0.01):
    '''
    读取off_path中的mesh，生成几何凹包，存入ply_path
    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(off_path)
    ms.generate_alpha_wrap(alpha_fraction=alpha, offset_fraction=offset)
    ms.save_current_mesh(ply_path)
    print(f"Generated alpha mesh: {ply_path}")
    return True

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
off_names = os.listdir(off_dir)
off_names.sort()

for i, off_name in enumerate(off_names):
    if not off_name.endswith('.off'):
        continue
        
    model_name = off_name.split('.')[0]
    print(f"Processing {model_name} ({i+1}/{len(off_names)})")
    
    # 源文件和目标文件路径
    off_path = os.path.join(off_dir, off_name)
    ply_path = os.path.join(ply_alpha_dir, f"{model_name}.ply")
    pc_path = os.path.join(pc_dir, f"{model_name}.csv")
    pc_noise_path = os.path.join(pc_noise_dir, f"{model_name}.csv")
    
    # 1. 从OFF生成alpha mesh，如果已经存在了就跳过
    if not os.path.exists(ply_path):
        generate_alpha_ply_from_off(off_path, ply_path, alpha=0.004, offset=0.003)
    
    # 2. 加载生成的mesh并进行缩放
    try:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        
        # 使用Open3D进行缩放，使高度标准化为100
        bbox = mesh.get_axis_aligned_bounding_box()
        height = bbox.get_max_bound()[2] - bbox.get_min_bound()[2]
        if height > 0:
            scale_factor = 100.0 / height
            mesh.scale(scale_factor, center=mesh.get_center())
            o3d.io.write_triangle_mesh(ply_path, mesh)  # 覆盖原文件
            # print(f"Scaled mesh to height 100, scale factor: {scale_factor}")
        
        # 打印三角面数、顶点数
        # print(f"Triangle number: {len(mesh.triangles)}, Vertex number: {len(mesh.vertices)}")
        # 可视化
        # o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

        # 3. 把mesh的顶点作为点云
        # pc = generate_pc_from_mesh(mesh, pc_num=sample_num, sigma=None)
        pc = np.asarray(mesh.vertices)
        
        # 4. 在pc基础上增加高斯噪声
        pc_noise = pc + np.random.normal(0, sigma, size=pc.shape)
        
        # 5. 保存点云
        np.savetxt(pc_path, pc, delimiter=',', fmt="%.6f")
        np.savetxt(pc_noise_path, pc_noise, delimiter=',', fmt="%.6f")
        
        print(f"Saved point clouds for {model_name}")
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

print("Done processing all models")