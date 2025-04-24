import os
import pandas as pd
import numpy as np

M = 'flexible'
DATA_DIR = f'datasets/ModelNet40_2/'  # 数据集根目录
ANALYZE2_DIR = f'dim_3/8_modelnet_2/analyze2_M{M}/'
ANALYZE5_DIR = f'dim_3/8_modelnet_2/analyze5_M{M}/'
os.makedirs(ANALYZE5_DIR, exist_ok=True)

# 初始化每种偏差的统计
bias_metrics = [
    # 注释掉的metric暂时不删除
    "Avg Distance to Shape", 
    "Chamfer Distance",
    # "Hausdorff Distance (Sample to GT)",
    # "Hausdorff Distance (GT to Sample)",
    "Hausdorff 95% Distance (Sample to GT)",
    "Hausdorff 95% Distance (GT to Sample)",
    # "Cloud to Cloud Distance (Sample to GT)",
    # "Cloud to Cloud Distance (GT to Sample)",
    "VFH Distance",
    # "Max Distance to Shape", 
    # "Min Distance to Shape",
]

metrics_show_name = {
    "Avg Distance to Shape": "pc_to_mesh_distance",
    "Chamfer Distance": "Chamfer_distance",
    "Hausdorff 95% Distance (Sample to GT)": "HD95_sample_to_gt",
    "Hausdorff 95% Distance (GT to Sample)": "HD95_gt_to_sample",
    "VFH Distance": "VFH_distance",
}

# 需要比较的方法
reference_method = "bds-hilbert"  # 使用bds-hilbert作为参考方法
comparison_methods = ['srs', 'fps', 'voxel', 'bds-pca']  # 需要与参考方法比较的方法
all_methods = comparison_methods + [reference_method] # 所有方法，包括参考方法

# 随机种子列表
random_seeds = [7, 42, 1309]

# 初始化统计
# 比值统计
method_ratios = {method: {metric: [] for metric in bias_metrics} for method in comparison_methods}

# 原始值统计
method_values = {method: {metric: [] for metric in bias_metrics} for method in all_methods}

# 大于参考方法的次数统计
method_greater_counts = {method: {metric: 0 for metric in bias_metrics} for method in comparison_methods}

# 实验对象总数
total_count = 0

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

    if data_id == 'keyboard_0001':
        continue
    
    # 增加实验对象计数
    total_count += 1

    # print(f"Processing ID: {data_id}")

    # 偏差比较
    bias_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id}_1st_bias.csv"
    if os.path.exists(bias_filename):
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
                        # 存储参考方法的原始值
                        method_values[reference_method][metric].append(reference_value)
            
            # 对于每个比较方法
            for method in comparison_methods:
                # 检查是否需要随机种子
                if method in ['srs', 'fps']:
                    # 对于需要随机种子的方法，先收集所有种子的值
                    method_values_temp = {metric: [] for metric in bias_metrics}
                    
                    # 收集所有种子的值
                    for seed in random_seeds:
                        method_with_seed = f"{method}_{seed}"
                        if method_with_seed in bias_df.columns:
                            for metric in bias_metrics:
                                if metric in bias_df.index:
                                    method_value = bias_df.loc[metric, method_with_seed]
                                    if pd.notna(method_value):
                                        method_values_temp[metric].append(method_value)
                    
                    # 计算每个指标的平均值
                    for metric, values in method_values_temp.items():
                        if values and metric in reference_values:
                            # 计算平均值
                            avg_value = np.mean(values)
                            reference_value = reference_values[metric]
                            
                            # 存储原始值
                            method_values[method][metric].append(avg_value)
                            
                            # 计算比值
                            if reference_value != 0:
                                ratio = avg_value / reference_value
                                method_ratios[method][metric].append(ratio)
                                
                                # 统计大于参考方法的次数
                                if avg_value > reference_value:
                                    method_greater_counts[method][metric] += 1
                else:
                    # 对于不需要随机种子的方法
                    if method in bias_df.columns:
                        # 计算比值
                        for metric, reference_value in reference_values.items():
                            if metric in bias_df.index:
                                method_value = bias_df.loc[metric, method]
                                if pd.notna(method_value) and reference_value != 0:
                                    # 存储原始值
                                    method_values[method][metric].append(method_value)
                                    
                                    ratio = method_value / reference_value
                                    method_ratios[method][metric].append(ratio)
                                    
                                    # 统计大于参考方法的次数
                                    if method_value > reference_value:
                                        method_greater_counts[method][metric] += 1
        else:
            print(f"\tWarning: {reference_method} not found in {bias_filename}")
    else:
        print(f"\tFile not found: {bias_filename}")

# 创建stats_ratio表格（比值统计）
# 列标题为各个方法名称加上后缀_mean
columns_ratio = []
for method in comparison_methods:
    columns_ratio.append(f"{method}_mean")

# 创建stats_ratio DataFrame
stats_ratio = pd.DataFrame(index=metrics_show_name.values(), columns=columns_ratio)

# 填充stats_ratio
for metric in bias_metrics:
    for method in comparison_methods:
        ratios = method_ratios[method][metric]
        if ratios:
            # 计算比值的平均值
            mean_ratio = np.mean(ratios)
            
            # 填充stats_ratio
            stats_ratio.loc[metrics_show_name[metric], f"{method}_mean"] = mean_ratio
        else:
            # 如果没有比值，填充NaN
            stats_ratio.loc[metrics_show_name[metric], f"{method}_mean"] = np.nan

# 保存比值统计结果
output_filename_ratio = f"{ANALYZE5_DIR}/exp4_stats_ratio.csv"
stats_ratio.to_csv(output_filename_ratio)
print(f"Saved ratio statistics to {output_filename_ratio}")

# 打印比值统计结果
print(f"\nRatio Statistics (Ratio to bds-hilbert):")
print(stats_ratio.to_string())

# 创建stats表格（原始值统计）
# 列标题为所有方法名称
columns_values = all_methods

# 创建stats DataFrame
stats_values = pd.DataFrame(index=metrics_show_name.values(), columns=columns_values)

# 填充stats_values
for metric in bias_metrics:
    for method in all_methods:
        values = method_values[method][metric]
        if values:
            # 计算原始值的平均值
            mean_value = np.mean(values)
            
            # 填充stats_values
            stats_values.loc[metrics_show_name[metric], method] = mean_value
        else:
            # 如果没有值，填充NaN
            stats_values.loc[metrics_show_name[metric], method] = np.nan

# 保存原始值统计结果
output_filename_values = f"{ANALYZE5_DIR}/exp4_stats.csv"
stats_values.to_csv(output_filename_values)
print(f"Saved value statistics to {output_filename_values}")

# 打印原始值统计结果
print(f"\nValue Statistics:")
print(stats_values.to_string())

# 创建stats_greater表格（大于参考方法的比例统计）
# 列标题为各个方法名称
columns_greater = comparison_methods

# 创建stats_greater DataFrame
stats_greater = pd.DataFrame(index=metrics_show_name.values(), columns=columns_greater)

# 填充stats_greater
for metric in bias_metrics:
    for method in comparison_methods:
        # 计算大于参考方法的比例
        greater_count = method_greater_counts[method][metric]
        
        if total_count > 0:
            greater_ratio = greater_count / total_count
            
            # 填充stats_greater
            stats_greater.loc[metrics_show_name[metric], method] = greater_ratio
        else:
            # 如果没有实验对象，填充NaN
            stats_greater.loc[metrics_show_name[metric], method] = np.nan

# 保存大于参考方法的比例统计结果
output_filename_greater = f"{ANALYZE5_DIR}/exp4_stats_greater.csv"
stats_greater.to_csv(output_filename_greater)
print(f"Saved greater than reference statistics to {output_filename_greater}")

# 打印大于参考方法的比例统计结果
print(f"\nGreater Than Reference Statistics:")
print(stats_greater.to_string())
