import os
import pandas as pd
import numpy as np

M = 'flexible'
DATA_DIR = f'dim_3/7_stanford_all/Stanford_all/'
PC_DIR = f"{DATA_DIR}pc/"
ANALYZE2_DIR = f'dim_3/7_stanford_all/analyze2_M{M}/'
ANALYZE3_DIR = f'dim_3/7_stanford_all/analyze3_M{M}/'
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
]


# 比较方法设置
bds_method = "bds-hilbert"  # BDS方法不需要随机种子
comparison_method = "fps_7"  # 使用voxel作为比较基准

# 初始化每个指标的总统计
total_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}

# 创建一个DataFrame来保存每个几何体的详细统计结果
geometry_stats = pd.DataFrame(columns=["Geometry", "Sample Ratio", "Metric", "BDS Lower", "BDS Equal", "BDS Higher", "Total", "Lower Ratio", "Equal Ratio", "Higher Ratio"])
stats_row = 0

# 从原始点云文件获取所有模型名称
model_names = [os.path.splitext(f)[0] for f in os.listdir(PC_DIR) if f.endswith('.csv')]
model_names.sort()  # 按字母顺序排序
print(f"Found {len(model_names)} models: {model_names}")

# 只处理10%采样比例
sample_ratio = 0.10
ratio_dir = f"{int(sample_ratio * 100)}"
print(f"Processing sample ratio: {sample_ratio}")

# 进行方法比较分析
ratio_folder_path = os.path.join(ANALYZE2_DIR, ratio_dir)
if not os.path.isdir(ratio_folder_path):
    print(f"Ratio folder not found: {ratio_folder_path}")
    exit(1)

# 遍历所有模型
for model_name in model_names:
    print(f"  Comparing methods for model: {model_name}")
    
    # 初始化当前模型的指标统计
    current_bias_results = {metric: {"BDS Lower": 0, "BDS Equal": 0, "BDS Higher": 0, "Total": 0} for metric in bias_metrics}

    # 偏差比较
    bias_filename = os.path.join(ratio_folder_path, f"{model_name}_1st_bias.csv")
    if not os.path.exists(bias_filename):
        print(f"    File not found: {bias_filename}")
        continue
        
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

    # 计算并保存当前模型的指标统计
    for metric in bias_metrics:
        total = current_bias_results[metric]["Total"]
        if total > 0:
            lower = current_bias_results[metric]["BDS Lower"]
            equal = current_bias_results[metric]["BDS Equal"]
            higher = current_bias_results[metric]["BDS Higher"]
            
            geometry_stats.loc[stats_row] = [
                model_name, 
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

# 保存几何体统计结果
output_filename = f"{ANALYZE3_DIR}/geometry_stats_{ratio_dir}.csv"
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
output_filename = f"{ANALYZE3_DIR}/total_stats_{ratio_dir}.csv"
total_stats.to_csv(output_filename, index=False)
print(f"Saved total statistics to {output_filename}")

# 打印总体统计结果
print("\n总偏差比例 (BDS Lower / Equal / Higher):")
print(total_stats.to_string(index=False))

print("\n分析完成!") 