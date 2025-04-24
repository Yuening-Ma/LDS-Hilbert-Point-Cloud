import os
import pandas as pd
import numpy as np

M = 'int'
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'
ANALYZE2_DIR = f'dim_3/5_closed_geometry_2/analyze2_M{M}/'
ANALYZE5_DIR = f'dim_3/5_closed_geometry_2/analyze5_M{M}/'
os.makedirs(ANALYZE5_DIR, exist_ok=True)

data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

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

# 定义数据组，N_9180表示这一组点云的点数为9180，id是0-14
data_groups = {
    'N_9180': {'min': 0, 'max': 14},
    'N_37240': {'min': 15, 'max': 29},
    'N_95760': {'min': 30, 'max': 44}
}

# 为每个组初始化统计
group_ratios = {group: {method: {metric: [] for metric in bias_metrics} for method in comparison_methods} 
                for group in data_groups.keys()}

# 为每个组初始化原始值统计
group_values = {group: {method: {metric: [] for metric in bias_metrics} for method in all_methods} 
                for group in data_groups.keys()}

# 为每个组初始化大于参考方法的次数统计
group_greater_counts = {group: {method: {metric: 0 for metric in bias_metrics} for method in comparison_methods} 
                        for group in data_groups.keys()}

# 为每个组初始化实验对象总数
group_total_counts = {group: 0 for group in data_groups.keys()}

for index, row in data_info.iterrows():
    data_id = int(row['ID'])
    geometry = row['Geometry']
    parameters = row['Parameters']
    data_points = row['DataPoints']

    # 确定数据属于哪个组
    current_group = None
    for group_name, group_range in data_groups.items():
        if group_range['min'] <= data_id <= group_range['max']:
            current_group = group_name
            break
    
    if current_group is None:
        print(f"Warning: Data ID {data_id} does not belong to any group")
        continue
    
    # 增加该组的实验对象计数
    group_total_counts[current_group] += 1

    # print(f"Processing ID: {data_id}, Geometry: {geometry}, Parameters: {parameters}, DataPoints: {data_points}")

    # for sample_ratio in [0.05, 0.10, 0.25]:
    for sample_ratio in [0.10]:
        # 偏差比较
        bias_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id:04}_1st_bias.csv"
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
                            group_values[current_group][reference_method][metric].append(reference_value)
                
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
                                
                                # 存储原始值
                                group_values[current_group][method][metric].append(avg_value)
                                
                                # 计算比值
                                if reference_value != 0:
                                    ratio = avg_value / reference_value
                                    group_ratios[current_group][method][metric].append(ratio)
                                    
                                    # 统计大于参考方法的次数
                                    if avg_value > reference_value:
                                        group_greater_counts[current_group][method][metric] += 1
                    else:
                        # 对于不需要随机种子的方法
                        if method in bias_df.columns:
                            # 计算比值
                            for metric, reference_value in reference_values.items():
                                if metric in bias_df.index:
                                    method_value = bias_df.loc[metric, method]
                                    if pd.notna(method_value) and reference_value != 0:
                                        # 存储原始值
                                        group_values[current_group][method][metric].append(method_value)
                                        
                                        ratio = method_value / reference_value
                                        group_ratios[current_group][method][metric].append(ratio)
                                        
                                        # 统计大于参考方法的次数
                                        if method_value > reference_value:
                                            group_greater_counts[current_group][method][metric] += 1
            else:
                print(f"\tWarning: {reference_method} not found in {bias_filename}")
        else:
            print(f"\tFile not found: {bias_filename}")

# 为每个组创建统计表
for group_name in data_groups.keys():
    # 创建group_stats_ratio表格（比值统计）
    # 列标题为各个方法名称加上后缀_mean
    columns_ratio = []
    for method in comparison_methods:
        columns_ratio.append(f"{method}_mean")
    
    # 创建group_stats_ratio DataFrame
    group_stats_ratio = pd.DataFrame(index=metrics_show_name.values(), columns=columns_ratio)
    
    # 填充group_stats_ratio
    for metric in bias_metrics:
        for method in comparison_methods:
            ratios = group_ratios[group_name][method][metric]
            if ratios:
                # 计算比值的平均值
                mean_ratio = np.mean(ratios)
                
                # 填充group_stats_ratio
                group_stats_ratio.loc[metrics_show_name[metric], f"{method}_mean"] = mean_ratio
            else:
                # 如果没有比值，填充NaN
                group_stats_ratio.loc[metrics_show_name[metric], f"{method}_mean"] = np.nan
    
    # 保存比值统计结果
    output_filename_ratio = f"{ANALYZE5_DIR}/exp3_{group_name}_stats_ratio.csv"
    group_stats_ratio.to_csv(output_filename_ratio)
    print(f"Saved {group_name} ratio statistics to {output_filename_ratio}")
    
    # 打印比值统计结果
    print(f"\n{group_name} Ratio Statistics (Ratio to bds-hilbert):")
    print(group_stats_ratio.to_string())
    
    # 创建group_stats表格（原始值统计）
    # 列标题为所有方法名称
    columns_values = all_methods
    
    # 创建group_stats DataFrame
    group_stats_values = pd.DataFrame(index=metrics_show_name.values(), columns=columns_values)
    
    # 填充group_stats_values
    for metric in bias_metrics:
        for method in all_methods:
            values = group_values[group_name][method][metric]
            if values:
                # 计算原始值的平均值
                mean_value = np.mean(values)
                
                # 填充group_stats_values
                group_stats_values.loc[metrics_show_name[metric], method] = mean_value
            else:
                # 如果没有值，填充NaN
                group_stats_values.loc[metrics_show_name[metric], method] = np.nan
    
    # 保存原始值统计结果
    output_filename_values = f"{ANALYZE5_DIR}/exp3_{group_name}_stats.csv"
    group_stats_values.to_csv(output_filename_values)
    print(f"Saved {group_name} value statistics to {output_filename_values}")
    
    # 打印原始值统计结果
    print(f"\n{group_name} Value Statistics:")
    print(group_stats_values.to_string())
    
    # 创建group_stats_greater表格（大于参考方法的比例统计）
    # 列标题为各个方法名称
    columns_greater = comparison_methods
    
    # 创建group_stats_greater DataFrame
    group_stats_greater = pd.DataFrame(index=metrics_show_name.values(), columns=columns_greater)
    
    # 填充group_stats_greater
    for metric in bias_metrics:
        for method in comparison_methods:
            # 计算大于参考方法的比例
            greater_count = group_greater_counts[group_name][method][metric]
            total_count = group_total_counts[group_name]
            
            if total_count > 0:
                greater_ratio = greater_count / total_count
                
                # 填充group_stats_greater
                group_stats_greater.loc[metrics_show_name[metric], method] = greater_ratio
            else:
                # 如果没有实验对象，填充NaN
                group_stats_greater.loc[metrics_show_name[metric], method] = np.nan
    
    # 保存大于参考方法的比例统计结果
    output_filename_greater = f"{ANALYZE5_DIR}/exp3_{group_name}_stats_greater.csv"
    group_stats_greater.to_csv(output_filename_greater)
    print(f"Saved {group_name} greater than reference statistics to {output_filename_greater}")
    
    # 打印大于参考方法的比例统计结果
    print(f"\n{group_name} Greater Than Reference Statistics:")
    print(group_stats_greater.to_string())

# 创建总表
# 1. 原始值总表
total_stats_df = pd.DataFrame()
for group_name in data_groups.keys():
    # 读取该组的统计结果
    group_stats = pd.read_csv(f"{ANALYZE5_DIR}/exp3_{group_name}_stats.csv", index_col=0)
    # 添加{N}_{sample_ratio}列
    group_stats['N'] = group_name.split('_')[1]
    # 添加到总表
    total_stats_df = pd.concat([total_stats_df, group_stats])

# 2. 比值总表
total_ratio_df = pd.DataFrame()
for group_name in data_groups.keys():
    # 读取该组的统计结果
    group_ratio = pd.read_csv(f"{ANALYZE5_DIR}/exp3_{group_name}_stats_ratio.csv", index_col=0)
    # 添加{N}_{sample_ratio}列
    group_ratio['N'] = group_name.split('_')[1]
    # 添加到总表
    total_ratio_df = pd.concat([total_ratio_df, group_ratio])

# 3. 大于参考方法的比例总表
total_greater_df = pd.DataFrame()
for group_name in data_groups.keys():
    # 读取该组的统计结果
    group_greater = pd.read_csv(f"{ANALYZE5_DIR}/exp3_{group_name}_stats_greater.csv", index_col=0)
    # 添加{N}_{sample_ratio}列
    group_greater['N'] = group_name.split('_')[1]
    # 添加到总表
    total_greater_df = pd.concat([total_greater_df, group_greater])

# 保存总表
total_stats_df.to_csv(f"{ANALYZE5_DIR}/exp3_total_stats.csv")
total_ratio_df.to_csv(f"{ANALYZE5_DIR}/exp3_total_stats_ratio.csv")
total_greater_df.to_csv(f"{ANALYZE5_DIR}/exp3_total_stats_greater.csv")

# 打印总表结果
print("\nTotal Original Value Statistics:")
print(total_stats_df.to_string())
print("\nTotal Ratio Statistics (Ratio to bds-hilbert):")
print(total_ratio_df.to_string())
print("\nTotal Greater Ratio Statistics:")
print(total_greater_df.to_string())
