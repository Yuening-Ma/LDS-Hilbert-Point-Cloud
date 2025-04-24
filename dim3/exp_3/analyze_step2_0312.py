import os
import pandas as pd
import numpy as np
import itertools

M = 'flexible'
DATA_DIR = f'datasets/ModelNet40_2/'  # 数据集根目录
ANALYZE1_DIR = f'dim_3/8_modelnet_2/analyze1_M{M}/'  # 第一次分析结果目录
ANALYZE2_DIR = f'dim_3/8_modelnet_2/analyze2_M{M}/'  # 第二次分析结果目录
SMOOTH_DIR = f'dim_3/8_modelnet_2/smooth_M{M}/'  # 平滑网格结果目录
os.makedirs(ANALYZE2_DIR, exist_ok=True)

random_seeds = [7, 42, 1309]
methods_need_seed = ['srs', 'fps']
methods_random_combinations = [f"{method}_{seed}" for method, seed in itertools.product(methods_need_seed, random_seeds)]
methods_all = methods_random_combinations + [
    'bds-pca', 
    'bds-hilbert', 
    'voxel'
]

# 遍历数据集根目录，提取，提取 data_id
data_ids = set()
for file_name in os.listdir(DATA_DIR + 'pc/'):
    if file_name.endswith('.csv'):
        data_id = file_name.split('.')[0]
        data_ids.add(data_id)

# for sample_ratio in [0.05, 0.10, 0.25]:
for sample_ratio in [0.20]:
    print(f"Sample Ratio: {sample_ratio}")
    os.makedirs(f'{ANALYZE2_DIR}{int(sample_ratio * 100)}/', exist_ok=True)
    
    for data_id in sorted(data_ids):
        print(f"  Processing data ID: {data_id}")

        '''
        对于所有降采样方法，提取第一个采样点云的指标
        '''
        # 初始化一个大的 DataFrame，用于存储所有方法的第一个采样点云的指标
        all_first_samples_df = pd.DataFrame(index=[
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
            "Smooth Min Distance to Shape"
        ])

        for method in methods_all:
            analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio * 100)}/{data_id}_{method}_analyze1.csv"
            if not os.path.exists(analyze1_filename):
                print(f"    File not found: {analyze1_filename}")
                continue

            analyze1_df = pd.read_csv(analyze1_filename, index_col=0)

            # 提取第一个采样点云的指标
            try:
                # 基本指标
                hausdorff_distance_sample_to_gt = analyze1_df.loc["Hausdorff Distance (Sample to GT)", "Sample_0"]
                hausdorff_distance_gt_to_sample = analyze1_df.loc["Hausdorff Distance (GT to Sample)", "Sample_0"]
                hausdorff_95_distance_sample_to_gt = analyze1_df.loc["Hausdorff 95% Distance (Sample to GT)", "Sample_0"]
                hausdorff_95_distance_gt_to_sample = analyze1_df.loc["Hausdorff 95% Distance (GT to Sample)", "Sample_0"]
                cloud_to_cloud_distance_sample_to_gt = analyze1_df.loc["Cloud to Cloud Distance (Sample to GT)", "Sample_0"]
                cloud_to_cloud_distance_gt_to_sample = analyze1_df.loc["Cloud to Cloud Distance (GT to Sample)", "Sample_0"]
                chamfer_distance = analyze1_df.loc["Chamfer Distance", "Sample_0"]
                vfh_distance = analyze1_df.loc["VFH Distance", "Sample_0"]
                avg_distance = analyze1_df.loc["Avg Distance to Shape", "Sample_0"]
                max_distance = analyze1_df.loc["Max Distance to Shape", "Sample_0"]
                min_distance = analyze1_df.loc["Min Distance to Shape", "Sample_0"]
                
                # 平滑网格指标
                smooth_hausdorff_distance_sample_to_gt = analyze1_df.loc["Smooth Hausdorff Distance (Sample to GT)", "Sample_0"]
                smooth_hausdorff_distance_gt_to_sample = analyze1_df.loc["Smooth Hausdorff Distance (GT to Sample)", "Sample_0"]
                smooth_hausdorff_95_distance_sample_to_gt = analyze1_df.loc["Smooth Hausdorff 95% Distance (Sample to GT)", "Sample_0"]
                smooth_hausdorff_95_distance_gt_to_sample = analyze1_df.loc["Smooth Hausdorff 95% Distance (GT to Sample)", "Sample_0"]
                smooth_cloud_to_cloud_distance_sample_to_gt = analyze1_df.loc["Smooth Cloud to Cloud Distance (Sample to GT)", "Sample_0"]
                smooth_cloud_to_cloud_distance_gt_to_sample = analyze1_df.loc["Smooth Cloud to Cloud Distance (GT to Sample)", "Sample_0"]
                smooth_chamfer_distance = analyze1_df.loc["Smooth Chamfer Distance", "Sample_0"]
                smooth_vfh_distance = analyze1_df.loc["Smooth VFH Distance", "Sample_0"]
                smooth_avg_distance = analyze1_df.loc["Smooth Avg Distance to Shape", "Sample_0"]
                smooth_max_distance = analyze1_df.loc["Smooth Max Distance to Shape", "Sample_0"]
                smooth_min_distance = analyze1_df.loc["Smooth Min Distance to Shape", "Sample_0"]
                
                # 将结果添加到 all_first_samples_df 中
                all_first_samples_df[method] = [
                    hausdorff_distance_sample_to_gt,
                    hausdorff_distance_gt_to_sample,
                    hausdorff_95_distance_sample_to_gt,
                    hausdorff_95_distance_gt_to_sample,
                    cloud_to_cloud_distance_sample_to_gt,
                    cloud_to_cloud_distance_gt_to_sample,
                    chamfer_distance,
                    vfh_distance,
                    avg_distance,
                    max_distance,
                    min_distance,
                    smooth_hausdorff_distance_sample_to_gt,
                    smooth_hausdorff_distance_gt_to_sample,
                    smooth_hausdorff_95_distance_sample_to_gt,
                    smooth_hausdorff_95_distance_gt_to_sample,
                    smooth_cloud_to_cloud_distance_sample_to_gt,
                    smooth_cloud_to_cloud_distance_gt_to_sample,
                    smooth_chamfer_distance,
                    smooth_vfh_distance,
                    smooth_avg_distance,
                    smooth_max_distance,
                    smooth_min_distance
                ]
            except KeyError as e:
                print(f"    Missing key in {analyze1_filename}: {e}")
                continue

        # 保存汇总的 all_first_samples_df 到文件
        output_filename = f"{ANALYZE2_DIR}{int(sample_ratio * 100)}/{data_id}_1st_bias.csv"
        all_first_samples_df.to_csv(output_filename, float_format="%.10f")
        print(f"    Saved: {output_filename}")

print("Done!")