import os
import pandas as pd
import numpy as np

M = 'flexible'
DATA_DIR = f'datasets/ModelNet40_2/'  # 数据集根目录
ANALYZE2_DIR = f'dim_3/8_modelnet_2/analyze2_M{M}/'
ANALYZE3_DIR = f'dim_3/8_modelnet_2/analyze3_M{M}/'
SMOOTH_DIR = f'dim_3/8_modelnet_2/smooth_M{M}/'  # 平滑网格结果目录
os.makedirs(ANALYZE3_DIR, exist_ok=True)

# 初始化每种偏差的统计
bias_metrics = [
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
]

# 比较方法设置
bds_method = "bds-hilbert"  # 可以根据需要修改为其他BDS方法
comparison_method = "fps_7"  # 可以根据需要修改为其他比较方法

# 初始化每个指标的总统计
total_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}

# 创建DataFrame存储每个几何体的详细统计
geometry_stats = pd.DataFrame(columns=["Geometry", "Sample Ratio"] + bias_metrics)
row_index = 0

# 遍历数据集根目录，提取 data_id
data_ids = set()
for file_name in os.listdir(DATA_DIR + 'pc/'):
    if file_name.endswith('.csv'):
        data_id = file_name.split('.')[0]
        data_ids.add(data_id)

# 只分析采样率为0.20的数据
sample_ratio = 0.20
print(f"Processing sample ratio: {sample_ratio}")

# 遍历每个数据集
for data_id in sorted(data_ids):

    # print(f"  Processing data ID: {data_id}")

    # numpy读取 f'{DATA_DIR}pc/data_id.csv'
    pc = np.loadtxt(f"{DATA_DIR}pc/{data_id}.csv", delimiter=',')
    N = pc.shape[0]

    # 初始化当前数据集和采样率的统计结果
    current_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}

    # 偏差比较
    bias_filename = f"{ANALYZE2_DIR}{int(sample_ratio * 100)}/{data_id}_1st_bias.csv"
    if os.path.exists(bias_filename):
        bias_df = pd.read_csv(bias_filename, index_col=0)
        
        # 检查BDS方法和比较方法是否存在于文件中
        if bds_method not in bias_df.columns or comparison_method not in bias_df.columns:
            print(f"    Missing methods in {bias_filename}")
            continue
            
        for metric in bias_metrics:
            # 检查指标是否存在于DataFrame中
            if metric not in bias_df.index:
                print(f"    Missing metric '{metric}' in {bias_filename}")
                continue
                
            bds_value = bias_df.loc[metric, bds_method]
            comparison_value = bias_df.loc[metric, comparison_method]
            
            # 检查是否两个值都有效
            if pd.notna(bds_value) and pd.notna(comparison_value):
                current_bias_results[metric]["Total"] += 1
                if bds_value < comparison_value:
                    current_bias_results[metric]["BDS Lower"] += 1
                elif bds_value == comparison_value:
                    current_bias_results[metric]["BDS Equal"] += 1
                else:
                    current_bias_results[metric]["BDS Higher"] += 1
                # 更新总统计
                total_bias_results[metric]["Total"] += 1
                if bds_value < comparison_value:
                    total_bias_results[metric]["BDS Lower"] += 1
                elif bds_value == comparison_value:
                    total_bias_results[metric]["BDS Equal"] += 1
                else:
                    total_bias_results[metric]["BDS Higher"] += 1
    else:
        print(f"    File not found: {bias_filename}")
        continue

    # 将当前几何体的结果添加到geometry_stats
    row_data = {"Geometry": data_id, "Sample Ratio": sample_ratio}
    for metric in bias_metrics:
        if current_bias_results[metric]["Total"] > 0:
            lower_ratio = current_bias_results[metric]["BDS Lower"] / current_bias_results[metric]["Total"]
            equal_ratio = current_bias_results[metric]["BDS Equal"] / current_bias_results[metric]["Total"]
            higher_ratio = current_bias_results[metric]["BDS Higher"] / current_bias_results[metric]["Total"]
            row_data[metric] = f"{lower_ratio:.3f},{equal_ratio:.3f},{higher_ratio:.3f}"
        else:
            row_data[metric] = "0.000,0.000,0.000"
    
    geometry_stats.loc[row_index] = row_data
    row_index += 1

# 保存几何体统计结果
geometry_stats_output = f"{ANALYZE3_DIR}geometry_stats_{int(sample_ratio * 100)}.csv"
geometry_stats.to_csv(geometry_stats_output, index=False)
print(f"Saved geometry statistics to: {geometry_stats_output}")

# 计算每个指标的总统计
total_stats = pd.DataFrame(index=bias_metrics, columns=["BDS Lower Ratio", "BDS Equal Ratio", "BDS Higher Ratio", "Total Count"])
for metric in bias_metrics:
    if total_bias_results[metric]["Total"] > 0:
        total_stats.loc[metric, "BDS Lower Ratio"] = total_bias_results[metric]["BDS Lower"] / total_bias_results[metric]["Total"]
        total_stats.loc[metric, "BDS Equal Ratio"] = total_bias_results[metric]["BDS Equal"] / total_bias_results[metric]["Total"]
        total_stats.loc[metric, "BDS Higher Ratio"] = total_bias_results[metric]["BDS Higher"] / total_bias_results[metric]["Total"]
        total_stats.loc[metric, "Total Count"] = total_bias_results[metric]["Total"]
    else:
        total_stats.loc[metric, "BDS Lower Ratio"] = 0.0
        total_stats.loc[metric, "BDS Equal Ratio"] = 0.0
        total_stats.loc[metric, "BDS Higher Ratio"] = 0.0
        total_stats.loc[metric, "Total Count"] = 0

# 保存总体统计结果
total_stats_output = f"{ANALYZE3_DIR}total_stats_{int(sample_ratio * 100)}.csv"
total_stats.to_csv(total_stats_output, float_format="%.3f")
print(f"Saved total statistics to: {total_stats_output}")

# 打印总体统计结果
print("\nTotal Statistics:")
print(total_stats.to_string())