import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')
from sample_evaluation import *

DATA_DIR = f'dim_3/4_surface_noised/data_Mint/'
SAMPLE_OUTPUT_DIR = f'dim_3/4_surface_noised/sample_Mint/'
ANALYZE1_DIR = f'dim_3/4_surface_noised/analyze1_Mint/'
os.makedirs(ANALYZE1_DIR, exist_ok=True)

data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

for index, row in data_info.iterrows():
    # 依次读取每一行的各个字段的值
    data_id = row['ID']
    surface = row['Surface']
    parameters = row['Parameters']
    data_points = row['DataPoints']

    # 打印读取的值
    print(f"ID: {data_id}, Surface: {surface}, Parameters: {parameters}, DataPoints: {data_points}")
    
    # 加载干净的点云（Ground Truth）
    pc_ground_truth = np.loadtxt(f"{DATA_DIR}{data_id:04}.csv", delimiter=',')
    # 加载加噪声的点云（Original）
    pc_original = np.loadtxt(f"{DATA_DIR}{data_id:04}_noised.csv", delimiter=',')

    N = pc_ground_truth.shape[0]

    assert N == data_points

    # 参数估计（真值）
    ground_truth_params = tuple(map(float, parameters.split('_')))
    if surface == "Plane":
        pc_params = estimate_and_test_plane(pc_original)
    elif surface == "Sphere":
        pc_params = estimate_and_test_sphere(pc_original)
    elif surface == "Ellipsoid":
        pc_params = estimate_and_test_ellipsoid(pc_original)
    elif surface == "Torus":
        pc_params = estimate_and_test_torus(pc_original)
    elif surface == "Cylinder":
        pc_params = estimate_and_test_cylinder(pc_original)
    elif surface == "Cone":
        pc_params = estimate_and_test_cone(pc_original)
    elif surface == "Paraboloid":
        pc_params = estimate_and_test_paraboloid(pc_original)

    # 计算原始点云的参数估计值
    original_params = pc_params[0]

    for sample_ratio in [0.10, 0.20]:
        num_sample = int(N * sample_ratio)
        print('\t', num_sample)

        os.makedirs(f'{ANALYZE1_DIR}{int(sample_ratio*100)}/', exist_ok=True)

        '''
        SRS, FPS
        '''
        for method in ['srs', 'fps']:
            for random_seed in [7, 42, 1309]:
                filename = f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_{random_seed}.csv'
                pc_sampled = np.loadtxt(filename, delimiter=',')

                # 第一行是每列的实际点数
                sample_sizes = pc_sampled[0, :].astype(int)
                pc_sampled = pc_sampled[1:, :]  # 去掉第一行

                # 创建 DataFrame
                df = pd.DataFrame(index=["Sample Size", 
                                         "Estimated Param 1", "Estimated Param 2", "Estimated Param 3", "Estimated Param 4",
                                         "Estimated Param 5", "Estimated Param 6",
                                         "Param 1 Diff (Ground Truth)", "Param 2 Diff (Ground Truth)", "Param 3 Diff (Ground Truth)", "Param 4 Diff (Ground Truth)",
                                         "Param 5 Diff (Ground Truth)", "Param 6 Diff (Ground Truth)",
                                         "Param 1 Diff (Original)", "Param 2 Diff (Original)", "Param 3 Diff (Original)", "Param 4 Diff (Original)",
                                         "Param 5 Diff (Original)", "Param 6 Diff (Original)",
                                         "KL Divergence (Ground Truth)", "KL Divergence (Original)",
                                         "JS Divergence (Ground Truth)", "JS Divergence (Original)",
                                         "R2", "RMSE"])  # 添加 R2 和 RMSE 行

                for i in range(0, pc_sampled.shape[1], 3):
                    sample = pc_sampled[:sample_sizes[i], i:i+3]  # 根据实际点数读取数据
                    if surface == "Plane":
                        results = estimate_and_test_plane(sample)
                    elif surface == "Sphere":
                        results = estimate_and_test_sphere(sample)
                    elif surface == "Ellipsoid":
                        results = estimate_and_test_ellipsoid(sample)
                    elif surface == "Torus":
                        results = estimate_and_test_torus(sample)
                    elif surface == "Cylinder":
                        results = estimate_and_test_cylinder(sample)
                    elif surface == "Cone":
                        results = estimate_and_test_cone(sample)
                    elif surface == "Paraboloid":
                        results = estimate_and_test_paraboloid(sample)

                    estimated_params = results[0]
                    r2 = results[1]  # 获取 R2
                    rmse = results[2]  # 获取 RMSE

                    # 计算参数差值
                    param_diffs_ground_truth = [abs(p1 - p2) for p1, p2 in zip(estimated_params, ground_truth_params)]
                    param_diffs_original = [abs(p1 - p2) for p1, p2 in zip(estimated_params, original_params)]

                    # 计算 KL 和 JS 散度
                    kl_ground_truth = kl_divergence_hist_3d(sample, pc_ground_truth)
                    kl_original = kl_divergence_hist_3d(sample, pc_original)
                    js_ground_truth = js_divergence_hist_3d(sample, pc_ground_truth)
                    js_original = js_divergence_hist_3d(sample, pc_original)

                    # 填充 DataFrame
                    df[f"Sample_{i//3}"] = [
                        sample_sizes[i],
                        estimated_params[0] if len(estimated_params) > 0 else None,
                        estimated_params[1] if len(estimated_params) > 1 else None,
                        estimated_params[2] if len(estimated_params) > 2 else None,
                        estimated_params[3] if len(estimated_params) > 3 else None,
                        estimated_params[4] if len(estimated_params) > 4 else None,
                        estimated_params[5] if len(estimated_params) > 5 else None,
                        param_diffs_ground_truth[0] if len(param_diffs_ground_truth) > 0 else None,
                        param_diffs_ground_truth[1] if len(param_diffs_ground_truth) > 1 else None,
                        param_diffs_ground_truth[2] if len(param_diffs_ground_truth) > 2 else None,
                        param_diffs_ground_truth[3] if len(param_diffs_ground_truth) > 3 else None,
                        param_diffs_ground_truth[4] if len(param_diffs_ground_truth) > 4 else None,
                        param_diffs_ground_truth[5] if len(param_diffs_ground_truth) > 5 else None,
                        param_diffs_original[0] if len(param_diffs_original) > 0 else None,
                        param_diffs_original[1] if len(param_diffs_original) > 1 else None,
                        param_diffs_original[2] if len(param_diffs_original) > 2 else None,
                        param_diffs_original[3] if len(param_diffs_original) > 3 else None,
                        param_diffs_original[4] if len(param_diffs_original) > 4 else None,
                        param_diffs_original[5] if len(param_diffs_original) > 5 else None,
                        kl_ground_truth,
                        kl_original,
                        js_ground_truth,
                        js_original,
                        r2,  # 添加 R2 结果
                        rmse  # 添加 RMSE 结果
                    ]

                # 保存到文件
                output_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_{random_seed}_analyze1.csv"
                df.to_csv(output_filename)
                print(f"\t\tSaved: {output_filename}")

        '''
        BDS, Voxel
        '''
        for method in [
            'bds-pca', 'bds-hilbert', 
            'voxel'
        ]:
            filename = f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}.csv'
            pc_sampled = np.loadtxt(filename, delimiter=',')

            # 第一行是每列的实际点数
            sample_sizes = pc_sampled[0, :].astype(int)
            pc_sampled = pc_sampled[1:, :]  # 去掉第一行

            # 创建 DataFrame
            df = pd.DataFrame(index=["Sample Size", 
                                     "Estimated Param 1", "Estimated Param 2", "Estimated Param 3", "Estimated Param 4",
                                     "Estimated Param 5", "Estimated Param 6",
                                     "Param 1 Diff (Ground Truth)", "Param 2 Diff (Ground Truth)", "Param 3 Diff (Ground Truth)", "Param 4 Diff (Ground Truth)",
                                     "Param 5 Diff (Ground Truth)", "Param 6 Diff (Ground Truth)",
                                     "Param 1 Diff (Original)", "Param 2 Diff (Original)", "Param 3 Diff (Original)", "Param 4 Diff (Original)",
                                     "Param 5 Diff (Original)", "Param 6 Diff (Original)",
                                     "KL Divergence (Ground Truth)", "KL Divergence (Original)",
                                     "JS Divergence (Ground Truth)", "JS Divergence (Original)",
                                     "R2", "RMSE"])  # 添加 R2 和 RMSE 行

            for i in range(0, pc_sampled.shape[1], 3):
                sample = pc_sampled[:sample_sizes[i], i:i+3]  # 根据实际点数读取数据

                if surface == "Plane":
                    results = estimate_and_test_plane(sample)
                elif surface == "Sphere":
                    results = estimate_and_test_sphere(sample)
                elif surface == "Ellipsoid":
                    results = estimate_and_test_ellipsoid(sample)
                elif surface == "Torus":
                    results = estimate_and_test_torus(sample)
                elif surface == "Cylinder":
                    results = estimate_and_test_cylinder(sample)
                elif surface == "Cone":
                    results = estimate_and_test_cone(sample)
                elif surface == "Paraboloid":
                    results = estimate_and_test_paraboloid(sample)

                estimated_params = results[0]
                r2 = results[1]  # 获取 R2
                rmse = results[2]  # 获取 RMSE

                # 计算参数差值
                param_diffs_ground_truth = [abs(p1 - p2) for p1, p2 in zip(estimated_params, ground_truth_params)]
                param_diffs_original = [abs(p1 - p2) for p1, p2 in zip(estimated_params, original_params)]

                # 计算 KL 和 JS 散度
                kl_ground_truth = kl_divergence_hist_3d(sample, pc_ground_truth)
                kl_original = kl_divergence_hist_3d(sample, pc_original)
                js_ground_truth = js_divergence_hist_3d(sample, pc_ground_truth)
                js_original = js_divergence_hist_3d(sample, pc_original)

                # 填充 DataFrame
                df[f"Sample_{i//3}"] = [
                    sample_sizes[i],
                    estimated_params[0] if len(estimated_params) > 0 else None,
                    estimated_params[1] if len(estimated_params) > 1 else None,
                    estimated_params[2] if len(estimated_params) > 2 else None,
                    estimated_params[3] if len(estimated_params) > 3 else None,
                    estimated_params[4] if len(estimated_params) > 4 else None,
                    estimated_params[5] if len(estimated_params) > 5 else None,
                    param_diffs_ground_truth[0] if len(param_diffs_ground_truth) > 0 else None,
                    param_diffs_ground_truth[1] if len(param_diffs_ground_truth) > 1 else None,
                    param_diffs_ground_truth[2] if len(param_diffs_ground_truth) > 2 else None,
                    param_diffs_ground_truth[3] if len(param_diffs_ground_truth) > 3 else None,
                    param_diffs_ground_truth[4] if len(param_diffs_ground_truth) > 4 else None,
                    param_diffs_ground_truth[5] if len(param_diffs_ground_truth) > 5 else None,
                    param_diffs_original[0] if len(param_diffs_original) > 0 else None,
                    param_diffs_original[1] if len(param_diffs_original) > 1 else None,
                    param_diffs_original[2] if len(param_diffs_original) > 2 else None,
                    param_diffs_original[3] if len(param_diffs_original) > 3 else None,
                    param_diffs_original[4] if len(param_diffs_original) > 4 else None,
                    param_diffs_original[5] if len(param_diffs_original) > 5 else None,
                    kl_ground_truth,
                    kl_original,
                    js_ground_truth,
                    js_original,
                    r2,  # 添加 R2 结果
                    rmse  # 添加 RMSE 结果
                ]

            # 保存到文件
            output_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"
            df.to_csv(output_filename)
            print(f"\t\tSaved: {output_filename}")