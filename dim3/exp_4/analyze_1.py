import os
import pandas as pd
import numpy as np
import sys
import open3d as o3d

# 添加自定义模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')
from sample_evaluation import *
from geometry_3d import *

M = 'flexible'
DATA_DIR = f'dim_3/7_stanford_all/Stanford_all/'  # 数据集根目录
PLY_DIR = f"{DATA_DIR}ply_scaled/"  # 缩放后的PLY文件目录
PC_DIR = f"{DATA_DIR}pc/"  # 原始点云目录
PC_NOISE_DIR = f"{DATA_DIR}pc_noise/"  # 带噪声点云目录
SAMPLE_OUTPUT_DIR = f'dim_3/7_stanford_all/sample_M{M}/'  # 采样结果目录
ANALYZE1_DIR = f'dim_3/7_stanford_all/analyze1_M{M}/'  # 分析结果目录
RECONSTRUCTED_DIR = f'dim_3/7_stanford_all/reconstructed_M{M}/'  # 重建网格目录
SMOOTH_DIR = f'dim_3/7_stanford_all/smooth_M{M}/'  # 平滑网格结果目录
os.makedirs(ANALYZE1_DIR, exist_ok=True)
os.makedirs(SMOOTH_DIR, exist_ok=True)
os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)  # 创建重建网格目录
BIN_NUM = 50
RADIUS = 5
K = 30
ball_pivoting_para = 27000

# 随机种子列表
random_seeds = [7, 42, 1309]

# 需要随机种子的方法
methods_need_seed = [
    'srs', 
    'fps'
]

# 不需要随机种子的方法
methods_no_seed = [
    'bds-pca',
    'bds-hilbert', 
    'voxel'
]

# 遍历 pc 文件夹中的所有 csv 文件
for file_name in sorted(os.listdir(PC_DIR)):
    if file_name.endswith('.csv'):
        # 获取文件名（不带扩展名）作为 data_id
        data_id = os.path.splitext(file_name)[0]

        # if not data_id.startswith('bun'):
        #     continue

        print(f"Processing file: {file_name}")

        # 加载干净的点云（Ground Truth）
        pc_ground_truth = np.loadtxt(f"{PC_DIR}{file_name}", delimiter=',')
        # 加载加噪声的点云（Original）
        pc_original = np.loadtxt(f"{PC_NOISE_DIR}{file_name}", delimiter=',')

        # 将 numpy 数组转换为 Open3D 点云对象
        pcd_ground_truth = o3d.geometry.PointCloud()
        pcd_ground_truth.points = o3d.utility.Vector3dVector(pc_ground_truth)

        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(pc_original)

        N = pc_ground_truth.shape[0]

        # 计算 VFH 特征 - 只计算ground truth的
        vfh_ground_truth = compute_VFH(pcd_ground_truth, k=max(3, round(N / 300)), temp_dir='temp')

        # 检查是否存在对应的PLY文件（用于点到表面距离计算）
        ply_path = f"{PLY_DIR}{data_id}.ply"
        has_mesh = os.path.exists(ply_path)

        # for sample_ratio in [0.05, 0.10, 0.25]:
        for sample_ratio in [0.10]:

            # 根据 data_id 开头的三个字母来设置不同的实际采样率
            if data_id.startswith('Arm'):
                actual_sample_ratio = 0.2
            elif data_id.startswith('dra'):
                actual_sample_ratio = 0.2
            else:
                actual_sample_ratio = 0.10

            num_sample = int(N * actual_sample_ratio)
            print(f'\tSample ratio: {actual_sample_ratio}, Sample count: {num_sample}')

            os.makedirs(f'{ANALYZE1_DIR}{int(sample_ratio * 100)}/', exist_ok=True)
            os.makedirs(f'{SMOOTH_DIR}{int(sample_ratio * 100)}/', exist_ok=True)
            os.makedirs(f'{RECONSTRUCTED_DIR}{int(sample_ratio * 100)}/', exist_ok=True)

            # 处理需要随机种子的方法
            for method in methods_need_seed:
                for random_seed in random_seeds:
                    filename = f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id}_{method}_{random_seed}.csv'
                    if not os.path.exists(filename):
                        print(f"\t\tFile not found: {filename}")
                        continue

                    pc_sampled = np.loadtxt(filename, delimiter=',')
                    sample_sizes = pc_sampled[0, :].astype(int)
                    pc_sampled = pc_sampled[1:, :]  # 去掉第一行

                    # 准备DataFrame索引 - 检查是否存在mesh
                    df = pd.DataFrame(index=["Sample Size", 
                                            "Hausdorff Distance (Sample to GT)",
                                            "Hausdorff Distance (GT to Sample)",
                                            "Hausdorff 95% Distance (Sample to GT)",
                                            "Hausdorff 95% Distance (GT to Sample)",
                                            "Cloud to Cloud Distance (Sample to GT)",
                                            "Cloud to Cloud Distance (GT to Sample)",
                                            "Chamfer Distance",
                                            "VFH Distance",
                                            "Avg Distance to Shape", 
                                            "Max Distance to Shape", 
                                            "Min Distance to Shape",
                                            "Smooth Hausdorff Distance (Sample to GT)",
                                            "Smooth Hausdorff Distance (GT to Sample)",
                                            "Smooth Hausdorff 95% Distance (Sample to GT)",
                                            "Smooth Hausdorff 95% Distance (GT to Sample)",
                                            "Smooth Cloud to Cloud Distance (Sample to GT)",
                                            "Smooth Cloud to Cloud Distance (GT to Sample)",
                                            "Smooth Chamfer Distance",
                                            "Smooth VFH Distance",
                                            "Smooth Avg Distance to Shape", 
                                            "Smooth Max Distance to Shape", 
                                            "Smooth Min Distance to Shape"])

                    for i in range(0, 3, 3):  # 只计算第一个群的结果
                        sample = pc_sampled[:sample_sizes[i], i:i+3]
                        pcd_sampled = o3d.geometry.PointCloud()
                        pcd_sampled.points = o3d.utility.Vector3dVector(sample)

                        # 使用pcd_distance函数计算点云距离指标
                        distance_metrics = pcd_distance(pcd_sampled, pcd_ground_truth)
                        hausdorff_dist_sample_to_gt = distance_metrics[0]
                        hausdorff_dist_gt_to_sample = distance_metrics[1]
                        hausdorff_95_dist_sample_to_gt = distance_metrics[2]
                        hausdorff_95_dist_gt_to_sample = distance_metrics[3]
                        cloud_to_cloud_dist_sample_to_gt = distance_metrics[4]
                        cloud_to_cloud_dist_gt_to_sample = distance_metrics[5]
                        chamfer_d = distance_metrics[6]

                        # 计算VFH特征
                        vfh_sampled = compute_VFH(pcd_sampled, k=max(3, round(sample.shape[0] / 300)), temp_dir='temp')
                        vfh_dist = VFH_distance(vfh_sampled, vfh_ground_truth)
                        
                        # 如果存在mesh，计算点到表面距离
                        if has_mesh:
                            sample_distances = cloud_to_mesh(sample, ply_path)
                            avg_distance = np.mean(sample_distances)
                            max_distance = np.max(sample_distances)
                            min_distance = np.min(sample_distances)
                        else:
                            avg_distance = np.nan
                            max_distance = np.nan
                            min_distance = np.nan

                        # 生成平滑网格并计算其指标
                        d = round((ball_pivoting_para / sample.shape[0]) ** 0.5, 1)
                        radii = [2 * d, 2.25 * d, 2.5 * d, 2.75 * d, 3 * d]

                        # 先尝试读取已有的mesh_reconstructed文件
                        reconstructed_path = f"{RECONSTRUCTED_DIR}{int(sample_ratio*100)}/{data_id}_{method}_{random_seed}.ply"
                        smooth_path = f"{SMOOTH_DIR}{int(sample_ratio*100)}/{data_id}_{method}_{random_seed}_smooth_mesh.ply"

                        # 尝试读取重建网格
                        if os.path.exists(reconstructed_path):
                            print(f"\t\t读取已有重建网格: {reconstructed_path}")
                            mesh_reconstructed = o3d.io.read_triangle_mesh(reconstructed_path)
                        
                            # 尝试读取平滑网格
                            if os.path.exists(smooth_path):
                                print(f"\t\t读取已有平滑网格: {smooth_path}")
                                smooth_mesh = o3d.io.read_triangle_mesh(smooth_path)
                            else:
                                print(f"\t\t平滑网格不存在: {smooth_path}")
                                smooth_mesh = None
                        else:
                            print(f"\t\t重建网格不存在，开始重建: {reconstructed_path}")
                            # 使用get_smooth_mesh重建网格
                            mesh_reconstructed, smooth_mesh = get_smooth_mesh(pcd_sampled, radius=RADIUS, k=K, radii=radii)
                        
                            # 保存重建网格
                            if mesh_reconstructed is not None:
                                o3d.io.write_triangle_mesh(reconstructed_path, mesh_reconstructed)
                                print(f"\t\t保存重建网格: {reconstructed_path}")
                            else:
                                o3d.io.write_triangle_mesh(reconstructed_path, o3d.geometry.TriangleMesh())
                                print(f"\t\t保存空重建网格: {reconstructed_path}")

                            # 保存平滑网格
                            if smooth_mesh is not None:
                                o3d.io.write_triangle_mesh(smooth_path, smooth_mesh)
                                print(f"\t\t保存平滑网格: {smooth_path}")
                            else:
                                o3d.io.write_triangle_mesh(smooth_path, o3d.geometry.TriangleMesh())
                                print(f"\t\t保存空平滑网格: {smooth_path}")
                        
                        # 初始化平滑网格指标为NaN
                        smooth_hausdorff_dist_sample_to_gt = np.nan
                        smooth_hausdorff_dist_gt_to_sample = np.nan
                        smooth_hausdorff_95_dist_sample_to_gt = np.nan
                        smooth_hausdorff_95_dist_gt_to_sample = np.nan
                        smooth_cloud_to_cloud_dist_sample_to_gt = np.nan
                        smooth_cloud_to_cloud_dist_gt_to_sample = np.nan
                        smooth_chamfer_distance = np.nan
                        smooth_vfh_dist = np.nan
                        smooth_avg_distance = np.nan
                        smooth_max_distance = np.nan
                        smooth_min_distance = np.nan
                        
                        # 如果平滑网格生成成功，计算指标
                        if smooth_mesh is not None and len(smooth_mesh.vertices) > 0:
                            # 将平滑网格的顶点转为点云
                            smooth_vertices = np.asarray(smooth_mesh.vertices)
                            pcd_smooth = o3d.geometry.PointCloud()
                            pcd_smooth.points = o3d.utility.Vector3dVector(smooth_vertices)
                            
                            # 计算平滑网格与ground truth的距离，使用pcd_distance函数计算
                            smooth_distance_metrics = pcd_distance(pcd_smooth, pcd_ground_truth)
                            smooth_hausdorff_dist_sample_to_gt = smooth_distance_metrics[0]
                            smooth_hausdorff_dist_gt_to_sample = smooth_distance_metrics[1]
                            smooth_hausdorff_95_dist_sample_to_gt = smooth_distance_metrics[2]
                            smooth_hausdorff_95_dist_gt_to_sample = smooth_distance_metrics[3]
                            smooth_cloud_to_cloud_dist_sample_to_gt = smooth_distance_metrics[4]
                            smooth_cloud_to_cloud_dist_gt_to_sample = smooth_distance_metrics[5]
                            smooth_chamfer_distance = smooth_distance_metrics[6]
                            vfh_smooth = compute_VFH(pcd_smooth, k=max(3, round(smooth_vertices.shape[0] / 300)), temp_dir='temp')
                            smooth_vfh_dist = VFH_distance(vfh_smooth, vfh_ground_truth)

                            # 如果存在mesh，计算平滑网格点到表面距离
                            if has_mesh:
                                smooth_distances = cloud_to_mesh(smooth_vertices, ply_path)
                                smooth_avg_distance = np.mean(smooth_distances)
                                smooth_max_distance = np.max(smooth_distances)
                                smooth_min_distance = np.min(smooth_distances)

                        # 根据是否有mesh填充结果
                        df[f"Sample_{i//3}"] = [
                            sample_sizes[i],
                            hausdorff_dist_sample_to_gt,
                            hausdorff_dist_gt_to_sample,
                            hausdorff_95_dist_sample_to_gt,
                            hausdorff_95_dist_gt_to_sample,
                            cloud_to_cloud_dist_sample_to_gt,
                            cloud_to_cloud_dist_gt_to_sample,
                            chamfer_d,
                            vfh_dist,
                            avg_distance,
                            max_distance,
                            min_distance,
                            smooth_hausdorff_dist_sample_to_gt,
                            smooth_hausdorff_dist_gt_to_sample,
                            smooth_hausdorff_95_dist_sample_to_gt,
                            smooth_hausdorff_95_dist_gt_to_sample,
                            smooth_cloud_to_cloud_dist_sample_to_gt,
                            smooth_cloud_to_cloud_dist_gt_to_sample,
                            smooth_chamfer_distance,
                            smooth_vfh_dist,
                            smooth_avg_distance,
                            smooth_max_distance,
                            smooth_min_distance
                        ]

                    output_filename = f"{ANALYZE1_DIR}{int(sample_ratio * 100)}/{data_id}_{method}_{random_seed}_analyze1.csv"
                    df.to_csv(output_filename)
                    print(f"\t\tSaved: {output_filename}")

            # 处理不需要随机种子的方法
            for method in methods_no_seed:
                filename = f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id}_{method}.csv'
                if not os.path.exists(filename):
                    print(f"\t\tFile not found: {filename}")
                    continue

                pc_sampled = np.loadtxt(filename, delimiter=',')
                sample_sizes = pc_sampled[0, :].astype(int)
                pc_sampled = pc_sampled[1:, :]  # 去掉第一行

                # 准备DataFrame索引 - 检查是否存在mesh
                df = pd.DataFrame(index=["Sample Size", 
                                        "Hausdorff Distance (Sample to GT)",
                                        "Hausdorff Distance (GT to Sample)",
                                        "Hausdorff 95% Distance (Sample to GT)",
                                        "Hausdorff 95% Distance (GT to Sample)",
                                        "Cloud to Cloud Distance (Sample to GT)",
                                        "Cloud to Cloud Distance (GT to Sample)",
                                        "Chamfer Distance",
                                        "VFH Distance",
                                        "Avg Distance to Shape", 
                                        "Max Distance to Shape", 
                                        "Min Distance to Shape",
                                        "Smooth Hausdorff Distance (Sample to GT)",
                                        "Smooth Hausdorff Distance (GT to Sample)",
                                        "Smooth Hausdorff 95% Distance (Sample to GT)",
                                        "Smooth Hausdorff 95% Distance (GT to Sample)",
                                        "Smooth Cloud to Cloud Distance (Sample to GT)",
                                        "Smooth Cloud to Cloud Distance (GT to Sample)",
                                        "Smooth Chamfer Distance",
                                        "Smooth VFH Distance",
                                        "Smooth Avg Distance to Shape", 
                                        "Smooth Max Distance to Shape", 
                                        "Smooth Min Distance to Shape"])


                for i in range(0, 3, 3):  # 只计算第一个群的结果
                    # if method == 'bds-hilbert':
                    #     sample = pc_sampled[:sample_sizes[i], i+3:i+6]
                    # else:
                    #     sample = pc_sampled[:sample_sizes[i], i:i+3]
                    sample = pc_sampled[:sample_sizes[i], i:i+3]
                    pcd_sampled = o3d.geometry.PointCloud()
                    pcd_sampled.points = o3d.utility.Vector3dVector(sample)

                    # 计算原始点云的指标，只与ground truth比较，使用pcd_distance函数计算
                    distance_metrics = pcd_distance(pcd_sampled, pcd_ground_truth)
                    hausdorff_dist_sample_to_gt = distance_metrics[0]
                    hausdorff_dist_gt_to_sample = distance_metrics[1]
                    hausdorff_95_dist_sample_to_gt = distance_metrics[2]
                    hausdorff_95_dist_gt_to_sample = distance_metrics[3]
                    cloud_to_cloud_dist_sample_to_gt = distance_metrics[4]
                    cloud_to_cloud_dist_gt_to_sample = distance_metrics[5]
                    chamfer_d = distance_metrics[6]
                    
                    # 计算VFH特征
                    vfh_sampled = compute_VFH(pcd_sampled, k=max(3, round(sample.shape[0] / 300)), temp_dir='temp')
                    vfh_dist = VFH_distance(vfh_sampled, vfh_ground_truth)
                    
                    # 如果存在mesh，计算点到表面距离
                    if has_mesh:
                        sample_distances = cloud_to_mesh(sample, ply_path)
                        avg_distance = np.mean(sample_distances)
                        max_distance = np.max(sample_distances)
                        min_distance = np.min(sample_distances)
                    else:
                        avg_distance = np.nan
                        max_distance = np.nan
                        min_distance = np.nan

                    d = round((ball_pivoting_para / sample.shape[0]) ** 0.5, 1)
                    radii = [2 * d, 2.25 * d, 2.5 * d, 2.75 * d, 3 * d]

                    # 先尝试读取已有的mesh_reconstructed文件
                    reconstructed_path = f"{RECONSTRUCTED_DIR}{int(sample_ratio*100)}/{data_id}_{method}.ply"
                    smooth_path = f"{SMOOTH_DIR}{int(sample_ratio*100)}/{data_id}_{method}_smooth_mesh.ply"

                    # 尝试读取重建网格
                    if os.path.exists(reconstructed_path):  
                        print(f"\t\t读取已有重建网格: {reconstructed_path}")
                        mesh_reconstructed = o3d.io.read_triangle_mesh(reconstructed_path)

                        # 尝试读取平滑网格
                        if os.path.exists(smooth_path):
                            print(f"\t\t读取已有平滑网格: {smooth_path}")
                            smooth_mesh = o3d.io.read_triangle_mesh(smooth_path)
                        else:
                            print(f"\t\t平滑网格不存在: {smooth_path}")
                            smooth_mesh = None
                    else:
                        print(f"\t\t重建网格不存在: {reconstructed_path}")
                        # 使用get_smooth_mesh重建网格
                        mesh_reconstructed, smooth_mesh = get_smooth_mesh(pcd_sampled, radius=RADIUS, k=K, radii=radii)

                        # 保存重建网格，如果mesh_reconstructed为None，则保存一个空的网格
                        if mesh_reconstructed is not None:
                            o3d.io.write_triangle_mesh(reconstructed_path, mesh_reconstructed)
                        else:
                            o3d.io.write_triangle_mesh(reconstructed_path, o3d.geometry.TriangleMesh())

                        # 保存平滑网格，如果smooth_mesh为None，则保存一个空的网格
                        if smooth_mesh is not None:
                            o3d.io.write_triangle_mesh(smooth_path, smooth_mesh)
                        else:
                            o3d.io.write_triangle_mesh(smooth_path, o3d.geometry.TriangleMesh())
                    
                    # 初始化平滑网格指标为NaN
                    smooth_hausdorff_dist_sample_to_gt = np.nan
                    smooth_hausdorff_dist_gt_to_sample = np.nan
                    smooth_hausdorff_95_dist_sample_to_gt = np.nan
                    smooth_hausdorff_95_dist_gt_to_sample = np.nan
                    smooth_cloud_to_cloud_dist_sample_to_gt = np.nan
                    smooth_cloud_to_cloud_dist_gt_to_sample = np.nan
                    smooth_chamfer_distance = np.nan
                    smooth_vfh_dist = np.nan
                    smooth_avg_distance = np.nan
                    smooth_max_distance = np.nan
                    smooth_min_distance = np.nan
                    
                    # 如果平滑网格生成成功，计算指标
                    if smooth_mesh is not None and len(smooth_mesh.vertices) > 0:
                        # 将平滑网格的顶点转为点云
                        smooth_vertices = np.asarray(smooth_mesh.vertices)
                        pcd_smooth = o3d.geometry.PointCloud()
                        pcd_smooth.points = o3d.utility.Vector3dVector(smooth_vertices)
                        
                        # 计算平滑网格与ground truth的距离，使用pcd_distance函数计算
                        smooth_distance_metrics = pcd_distance(pcd_smooth, pcd_ground_truth)
                        smooth_hausdorff_dist_sample_to_gt = smooth_distance_metrics[0]
                        smooth_hausdorff_dist_gt_to_sample = smooth_distance_metrics[1]
                        smooth_hausdorff_95_dist_sample_to_gt = smooth_distance_metrics[2]
                        smooth_hausdorff_95_dist_gt_to_sample = smooth_distance_metrics[3]
                        smooth_cloud_to_cloud_dist_sample_to_gt = smooth_distance_metrics[4]
                        smooth_cloud_to_cloud_dist_gt_to_sample = smooth_distance_metrics[5]
                        smooth_chamfer_distance = smooth_distance_metrics[6]

                        # 计算VFH特征
                        vfh_smooth = compute_VFH(pcd_smooth, k=max(3, round(smooth_vertices.shape[0] / 300)), temp_dir='temp')
                        smooth_vfh_dist = VFH_distance(vfh_smooth, vfh_ground_truth)
                        
                        # 如果存在mesh，计算平滑网格点到表面距离
                        if has_mesh:
                            smooth_distances = cloud_to_mesh(smooth_vertices, ply_path)
                            smooth_avg_distance = np.mean(smooth_distances)
                            smooth_max_distance = np.max(smooth_distances)
                            smooth_min_distance = np.min(smooth_distances)

                    # 根据是否有mesh填充结果
                    df[f"Sample_{i//3}"] = [
                        sample_sizes[i],
                        hausdorff_dist_sample_to_gt,
                        hausdorff_dist_gt_to_sample,
                        hausdorff_95_dist_sample_to_gt,
                        hausdorff_95_dist_gt_to_sample,
                        cloud_to_cloud_dist_sample_to_gt,
                        cloud_to_cloud_dist_gt_to_sample,
                        chamfer_d,
                        vfh_dist,
                        avg_distance,
                        max_distance,
                        min_distance,
                        smooth_hausdorff_dist_sample_to_gt,
                        smooth_hausdorff_dist_gt_to_sample,
                        smooth_hausdorff_95_dist_sample_to_gt,
                        smooth_hausdorff_95_dist_gt_to_sample,
                        smooth_cloud_to_cloud_dist_sample_to_gt,
                        smooth_cloud_to_cloud_dist_gt_to_sample,
                        smooth_chamfer_distance,
                        smooth_vfh_dist,
                        smooth_avg_distance,
                        smooth_max_distance,
                        smooth_min_distance
                    ]

                output_filename = f"{ANALYZE1_DIR}{int(sample_ratio * 100)}/{data_id}_{method}_analyze1.csv"
                df.to_csv(output_filename)
                print(f"\t\tSaved: {output_filename}")

print("Analysis completed!") 