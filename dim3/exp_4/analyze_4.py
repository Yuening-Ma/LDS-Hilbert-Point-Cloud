import os
import pandas as pd
import numpy as np

M = 'flexible'
DATA_DIR = f'dim_3/7_stanford_all/Stanford_all/'
PC_DIR = f"{DATA_DIR}pc/"
ANALYZE2_DIR = f'dim_3/7_stanford_all/analyze2_M{M}/'
ANALYZE3_DIR = f'dim_3/7_stanford_all/analyze3_M{M}/'
ANALYZE4_DIR = f'dim_3/7_stanford_all/analyze4_M{M}/'
os.makedirs(ANALYZE4_DIR, exist_ok=True)

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
    # "Max Distance to Shape", 
    # "Min Distance to Shape",
]

# 需要比较的方法
reference_method = "bds-hilbert"  # 使用bds-hilbert作为参考方法
comparison_methods = ['srs', 'fps', 'voxel', 'bds-pca']  # 需要与参考方法比较的方法

# 随机种子列表
random_seeds = [7, 42, 1309]

# 初始化每个指标的总统计
# 对于每个比较方法，存储其与参考方法的比值
method_ratios = {method: {metric: [] for metric in bias_metrics} for method in comparison_methods}

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
    
    # 偏差比较
    bias_filename = os.path.join(ratio_folder_path, f"{model_name}_1st_bias.csv")
    if not os.path.exists(bias_filename):
        print(f"    File not found: {bias_filename}")
        continue
        
    bias_df = pd.read_csv(bias_filename, index_col=0)
    
    # 检查参考方法是否存在
    if reference_method in bias_df.columns:
        # 获取参考方法的值
        reference_values = {}
        for metric in bias_metrics:
            if metric in bias_df.index:
                reference_value = bias_df.loc[metric, reference_method]
                if pd.notna(reference_value):
                    reference_values[metric] = reference_value
        
        # 对于每个比较方法
        for method in comparison_methods:
            # 检查是否需要随机种子
            if method in ['srs', 'fps']:
                # 对于需要随机种子的方法，先收集所有种子的值
                method_values = {metric: [] for metric in bias_metrics}
                
                # 收集所有种子的值
                for seed in random_seeds:
                    method_with_seed = f"{method}_{seed}"
                    if method_with_seed in bias_df.columns:
                        for metric in bias_metrics:
                            if metric in bias_df.index:
                                method_value = bias_df.loc[metric, method_with_seed]
                                if pd.notna(method_value):
                                    method_values[metric].append(method_value)
                
                # 计算每个指标的平均值
                for metric, values in method_values.items():
                    if values and metric in reference_values:
                        # 计算平均值
                        avg_value = np.mean(values)
                        reference_value = reference_values[metric]
                        
                        # 计算比值
                        if reference_value != 0:
                            ratio = avg_value / reference_value
                            method_ratios[method][metric].append(ratio)
            else:
                # 对于不需要随机种子的方法
                if method in bias_df.columns:
                    # 计算比值
                    for metric, reference_value in reference_values.items():
                        if metric in bias_df.index:
                            method_value = bias_df.loc[metric, method]
                            if pd.notna(method_value) and reference_value != 0:
                                ratio = method_value / reference_value
                                method_ratios[method][metric].append(ratio)
    else:
        print(f"    Warning: {reference_method} not found in {bias_filename}")

# 创建total_stats表格
# 列标题为各个方法名称加上后缀_num和_mean
# 先添加所有的_num列，再添加所有的_mean列
columns = []
# 添加所有的_num列
for method in comparison_methods:
    columns.append(f"{method}_num")
# 添加所有的_mean列
for method in comparison_methods:
    columns.append(f"{method}_mean")

# 创建total_stats DataFrame
total_stats = pd.DataFrame(index=bias_metrics, columns=columns)

# 填充total_stats
for metric in bias_metrics:
    for method in comparison_methods:
        ratios = method_ratios[method][metric]
        if ratios:
            # 计算比值大于1的比例
            greater_than_one = sum(1 for ratio in ratios if ratio > 1)
            ratio_greater_than_one = greater_than_one / len(ratios)
            
            # 计算比值的平均值
            mean_ratio = np.mean(ratios)
            
            # 填充total_stats
            total_stats.loc[metric, f"{method}_num"] = ratio_greater_than_one
            total_stats.loc[metric, f"{method}_mean"] = mean_ratio
        else:
            # 如果没有比值，填充NaN
            total_stats.loc[metric, f"{method}_num"] = np.nan
            total_stats.loc[metric, f"{method}_mean"] = np.nan

# 保存总体统计结果
output_filename = f"{ANALYZE4_DIR}/exp5_total_stats_ratio.csv"
total_stats.to_csv(output_filename)
print(f"Saved total statistics to {output_filename}")

# 打印总体统计结果
print("\nTotal Statistics (Ratio to bds-hilbert):")
print(total_stats.to_string())