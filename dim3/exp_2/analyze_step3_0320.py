import os
import pandas as pd

M = 'int'
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'
ANALYZE2_DIR = f'dim_3/5_closed_geometry_2/analyze2_M{M}/'
ANALYZE3_DIR = f'dim_3/5_closed_geometry_2/analyze3_M{M}/'
os.makedirs(ANALYZE3_DIR, exist_ok=True)

data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

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
    "Smooth Min Distance to Shape",
]

# 需要比较的方法
bds_method = "bds-hilbert"
comparison_method = "fps_7"  # 使用fps_7作为比较基准

# 初始化每个指标的总统计
total_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}

# 创建一个DataFrame来保存每个几何体的详细统计结果
geometry_stats = pd.DataFrame(columns=["Geometry", "Sample Ratio", "Metric", "BDS Lower", "BDS Equal", "BDS Higher", "Total", "Lower Ratio", "Equal Ratio", "Higher Ratio"])
stats_row = 0

for index, row in data_info.iterrows():
    data_id = row['ID']
    geometry = row['Geometry']
    parameters = row['Parameters']
    data_points = row['DataPoints']

    # print(f"Processing ID: {data_id}, Geometry: {geometry}, Parameters: {parameters}, DataPoints: {data_points}")

    for sample_ratio in [0.10]:
        # 初始化当前数据集的指标统计
        current_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}

        # 偏差比较
        bias_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id:04}_1st_bias.csv"
        if os.path.exists(bias_filename):
            bias_df = pd.read_csv(bias_filename, index_col=0)
            
            # 检查必要的列是否存在
            if bds_method in bias_df.columns and comparison_method in bias_df.columns:
                for metric in bias_metrics:
                    # 检查当前指标是否在DataFrame的索引中
                    if metric in bias_df.index:
                        bds_value = bias_df.loc[metric, bds_method]
                        comparison_value = bias_df.loc[metric, comparison_method]
                        
                        # 检查是否两个值都有效
                        if pd.notna(bds_value) and pd.notna(comparison_value):
                            current_bias_results[metric]["Total"] += 1
                            total_bias_results[metric]["Total"] += 1
                            
                            if bds_value < comparison_value:
                                current_bias_results[metric]["BDS Lower"] += 1
                                total_bias_results[metric]["BDS Lower"] += 1
                            elif bds_value == comparison_value:
                                current_bias_results[metric]["BDS Equal"] += 1
                                total_bias_results[metric]["BDS Equal"] += 1
                            else:
                                current_bias_results[metric]["BDS Higher"] += 1
                                total_bias_results[metric]["BDS Higher"] += 1
            else:
                print(f"\tWarning: {bds_method} or {comparison_method} not found in {bias_filename}")
        else:
            print(f"\tFile not found: {bias_filename}")

        # 计算并保存当前数据集的指标统计
        for metric in bias_metrics:
            total = current_bias_results[metric]["Total"]
            if total > 0:
                lower = current_bias_results[metric]["BDS Lower"]
                equal = current_bias_results[metric]["BDS Equal"]
                higher = current_bias_results[metric]["BDS Higher"]
                
                geometry_stats.loc[stats_row] = [
                    geometry, 
                    sample_ratio,
                    metric,
                    lower,
                    equal,
                    higher,
                    total,
                    lower/total,
                    equal/total,
                    higher/total
                ]
                stats_row += 1
                
                # print(f"\t{metric}: {lower}/{total} lower, {equal}/{total} equal, {higher}/{total} higher")

# 保存几何体统计结果
output_filename = f"{ANALYZE3_DIR}/geometry_stats.csv"
geometry_stats.to_csv(output_filename, index=False)
print(f"Saved geometry statistics to {output_filename}")

# 计算每个指标的总体统计
total_stats = pd.DataFrame(columns=["Metric", "BDS Lower", "BDS Equal", "BDS Higher", "Total", "Lower Ratio", "Equal Ratio", "Higher Ratio"])
row = 0

for metric, result in total_bias_results.items():
    total = result["Total"]
    if total > 0:
        lower = result["BDS Lower"]
        equal = result["BDS Equal"]
        higher = result["BDS Higher"]
        
        total_stats.loc[row] = [
            metric,
            lower,
            equal,
            higher,
            total,
            lower/total,
            equal/total,
            higher/total
        ]
        row += 1

# 保存总体统计结果
output_filename = f"{ANALYZE3_DIR}/total_stats.csv"
total_stats.to_csv(output_filename, index=False)
print(f"Saved total statistics to {output_filename}")

# 打印总体统计结果
print("\nTotal Statistics:")
print(total_stats.to_string())