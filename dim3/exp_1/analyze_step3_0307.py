import os
import pandas as pd

DATA_DIR = f'dim_3/4_surface_noised/data_Mint/'
ANALYZE2_DIR = f'dim_3/4_surface_noised/analyze2_Mint/'
ANALYZE3_DIR = f'dim_3/4_surface_noised/analyze3_Mint/'
os.makedirs(ANALYZE3_DIR, exist_ok=True)

data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 初始化每种偏差和方差的统计
bias_metrics = [
    "Param 1 Diff (Ground Truth) Mean", 
    "Param 2 Diff (Ground Truth) Mean", 
    "Param 3 Diff (Ground Truth) Mean", 
    "Param 4 Diff (Ground Truth) Mean", 
    "Param 5 Diff (Ground Truth) Mean", 
    "Param 6 Diff (Ground Truth) Mean", 
    "Param 1 Diff (Original) Mean", 
    "Param 2 Diff (Original) Mean", 
    "Param 3 Diff (Original) Mean", 
    "Param 4 Diff (Original) Mean", 
    "Param 5 Diff (Original) Mean", 
    "Param 6 Diff (Original) Mean", 
]
variance_metrics = [
    "Param 1 Variance", 
    "Param 2 Variance", 
    "Param 3 Variance", 
    "Param 4 Variance", 
    "Param 5 Variance", 
    "Param 6 Variance"
]

# 初始化每个指标的总统计
total_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}
total_variance_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in variance_metrics}

for index, row in data_info.iterrows():
    data_id = row['ID']
    surface = row['Surface']  # 获取曲面信息
    parameters = row['Parameters']  # 获取参数信息
    data_points = row['DataPoints']  # 获取数据点数量

    for sample_ratio in [0.05, 0.10, 0.25]:
        # 初始化当前数据集和采样率的统计结果
        current_bias_bds_lower = 0
        current_bias_bds_equal = 0
        current_bias_bds_higher = 0
        current_bias_total = 0
        current_variance_bds_lower = 0
        current_variance_bds_equal = 0
        current_variance_bds_higher = 0
        current_variance_total = 0

        # 偏差比较
        bias_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id:04}_all_biases.csv"
        if os.path.exists(bias_filename):
            bias_df = pd.read_csv(bias_filename, index_col=0)
            for metric in bias_metrics:
                bds_value = bias_df.loc[metric, "bds-hilbert"]
                fps_value = bias_df.loc[metric, "fps_1309"]
                # 检查是否两个值都有效
                if pd.notna(bds_value) and pd.notna(fps_value):
                    current_bias_total += 1
                    if bds_value < fps_value:
                        current_bias_bds_lower += 1
                    elif bds_value == fps_value:
                        current_bias_bds_equal += 1
                    else:
                        current_bias_bds_higher += 1
                    # 更新总统计
                    total_bias_results[metric]["Total"] += 1
                    if bds_value < fps_value:
                        total_bias_results[metric]["BDS Lower"] += 1
                    elif bds_value == fps_value:
                        total_bias_results[metric]["BDS Equal"] += 1
                    else:
                        total_bias_results[metric]["BDS Higher"] += 1

        # 方差比较
        variance_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id:04}_cluster_variances.csv"
        if os.path.exists(variance_filename):
            variance_df = pd.read_csv(variance_filename, index_col=0)
            for metric in variance_metrics:
                bds_value = variance_df.loc[metric, "bds-hilbert"]
                fps_value = variance_df.loc[metric, "srs_7"]
                
                # 检查是否两个值都有效
                if pd.notna(bds_value) and pd.notna(fps_value):
                    current_variance_total += 1
                    if bds_value < fps_value:
                        current_variance_bds_lower += 1
                    elif bds_value == fps_value:
                        current_variance_bds_equal += 1
                    else:
                        current_variance_bds_higher += 1
                    # 更新总统计
                    total_variance_results[metric]["Total"] += 1
                    if bds_value < fps_value:
                        total_variance_results[metric]["BDS Lower"] += 1
                    elif bds_value == fps_value:
                        total_variance_results[metric]["BDS Equal"] += 1
                    else:
                        total_variance_results[metric]["BDS Higher"] += 1

        # 计算当前数据集和采样率的 BDS 更低的比例
        if current_bias_total > 0:
            bias_proportion = f"{current_bias_bds_lower}/{current_bias_total}, {current_bias_bds_lower / current_bias_total:.3f} {current_bias_bds_equal / current_bias_total:.3f} {current_bias_bds_higher / current_bias_total:.3f}"
        else:
            bias_proportion = "0/0, 0.000 0.000 0.000"

        if current_variance_total > 0:
            variance_proportion = f"{current_variance_bds_lower}/{current_variance_total}, {current_variance_bds_lower / current_variance_total:.3f} {current_variance_bds_equal / current_variance_total:.3f} {current_variance_bds_higher / current_variance_total:.3f}"
        else:
            variance_proportion = "0/0, 0.000 0.000 0.000"

# 计算每个指标的 BDS 更低的次数占总次数的均值
total_bias_proportions = {metric: f"{result['BDS Lower'] / result['Total']:.3f} {result['BDS Equal'] / result['Total']:.3f} {result['BDS Higher'] / result['Total']:.3f}" if result["Total"] > 0 else "0.000 0.000 0.000" for metric, result in total_bias_results.items()}
total_variance_proportions = {metric: f"{result['BDS Lower'] / result['Total']:.3f} {result['BDS Equal'] / result['Total']:.3f} {result['BDS Higher'] / result['Total']:.3f}" if result["Total"] > 0 else "0.000 0.000 0.000" for metric, result in total_variance_results.items()}

# 打印每个指标的总均值
print("\nTotal Bias Proportions (BDS Lower Mean):")
for metric, proportion in total_bias_proportions.items():
    print(f"{metric}: {proportion}")

print("\nTotal Variance Proportions (BDS Lower Mean):")
for metric, proportion in total_variance_proportions.items():
    print(f"{metric}: {proportion}")