import os
import pandas as pd
import numpy as np

M = 'int'
DATA_DIR = f'dim_3/4_surface_noised/data_M{M}/'
ANALYZE1_DIR = f'dim_3/4_surface_noised/analyze1_M{M}/'
ANALYZE5_DIR = f'dim_3/4_surface_noised/analyze5_M{M}/'
os.makedirs(ANALYZE5_DIR, exist_ok=True)

data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 需要比较的方法
reference_method = "bds-hilbert"  # 使用bds-hilbert作为参考方法
comparison_methods = ['srs', 'fps', 'voxel', 'bds-pca']  # 需要与参考方法比较的方法
all_methods = comparison_methods + [reference_method]  # 所有方法，包括参考方法

# 随机种子列表
random_seeds = [7, 42, 1309]

# 定义数据组，根据DataPoints分组
data_groups = {
    '9180': {'min': 0, 'max': 6},  # 前7个数据点数为9180
    '37240': {'min': 7, 'max': 13},  # 中间7个数据点数为37240
    '95760': {'min': 14, 'max': 20}  # 最后7个数据点数为95760
}

# 采样比例
sample_ratios = [0.10, 0.20]  # 可以添加更多采样比例，如[0.05, 0.10, 0.25]

# 初始化统计数据结构
# 使用字典存储每个{N}_{sample_ratio}组合的统计结果
stats_data = {}
ratio_data = {}
greater_data = {}

# 初始化每个组合的统计
for n in data_groups.keys():
    for ratio in sample_ratios:
        key = f"{n}_{int(ratio*100)}"
        # 初始化每个方法的统计列表
        stats_data[key] = {method: [] for method in all_methods}
        ratio_data[key] = {method: [] for method in comparison_methods}
        greater_data[key] = {method: 0 for method in comparison_methods}

# 遍历所有数据
for index, row in data_info.iterrows():
    data_id = int(row['ID'])
    surface = row['Surface']
    parameters = row['Parameters']
    data_points = row['DataPoints']

    # 确定数据属于哪个组
    current_n = None
    for n, group_range in data_groups.items():
        if group_range['min'] <= data_id <= group_range['max']:
            current_n = n
            break
    
    if current_n is None:
        print(f"Warning: Data ID {data_id} does not belong to any group")
        continue

    # 处理每个采样比例
    for sample_ratio in sample_ratios:
        key = f"{current_n}_{int(sample_ratio*100)}"
        
        # 计算每个方法的L1距离
        for method in all_methods:
            # 对于需要随机种子的方法，计算所有种子的平均值
            if method in ['srs', 'fps']:
                method_values = []
                for seed in random_seeds:
                    # 读取analyze1的结果文件
                    analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_{seed}_analyze1.csv"
                    if not os.path.exists(analyze1_filename):
                        print(f"File not found: {analyze1_filename}")
                        continue

                    analyze1_df = pd.read_csv(analyze1_filename, index_col=0)
                    
                    # 计算该种子的L1距离
                    l1_distance = 0
                    for i in range(1, 7):  # 最多6个参数
                        param_diff = analyze1_df.loc[f"Param {i} Diff (Ground Truth)", "Sample_0"]
                        if pd.notna(param_diff):
                            l1_distance += abs(param_diff)
                    method_values.append(l1_distance)
                
                if method_values:
                    avg_l1_distance = np.mean(method_values)
                    stats_data[key][method].append(avg_l1_distance)
            else:
                # 对于不需要随机种子的方法
                analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"
                if not os.path.exists(analyze1_filename):
                    print(f"File not found: {analyze1_filename}")
                    continue

                analyze1_df = pd.read_csv(analyze1_filename, index_col=0)
                
                # 计算L1距离
                l1_distance = 0
                for i in range(1, 7):  # 最多6个参数
                    param_diff = analyze1_df.loc[f"Param {i} Diff (Ground Truth)", "Sample_0"]
                    if pd.notna(param_diff):
                        l1_distance += abs(param_diff)
                stats_data[key][method].append(l1_distance)

        # 计算比值和大于参考方法的次数
        # 读取参考方法的结果文件
        ref_analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{reference_method}_analyze1.csv"
        if not os.path.exists(ref_analyze1_filename):
            print(f"File not found: {ref_analyze1_filename}")
            continue

        ref_analyze1_df = pd.read_csv(ref_analyze1_filename, index_col=0)
        
        # 计算参考方法的L1距离
        ref_l1_distance = 0
        for i in range(1, 7):
            param_diff = ref_analyze1_df.loc[f"Param {i} Diff (Ground Truth)", "Sample_0"]
            if pd.notna(param_diff):
                ref_l1_distance += abs(param_diff)
        
        # 对于每个比较方法
        for method in comparison_methods:
            if method in ['srs', 'fps']:
                # 计算所有种子的平均L1距离
                method_values = []
                for seed in random_seeds:
                    analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_{seed}_analyze1.csv"
                    if not os.path.exists(analyze1_filename):
                        continue

                    analyze1_df = pd.read_csv(analyze1_filename, index_col=0)
                    
                    l1_distance = 0
                    for i in range(1, 7):
                        param_diff = analyze1_df.loc[f"Param {i} Diff (Ground Truth)", "Sample_0"]
                        if pd.notna(param_diff):
                            l1_distance += abs(param_diff)
                    method_values.append(l1_distance)
                
                if method_values:
                    avg_l1_distance = np.mean(method_values)
                    # 计算比值
                    if ref_l1_distance != 0:
                        ratio = avg_l1_distance / ref_l1_distance
                        ratio_data[key][method].append(ratio)
                        # 统计大于参考方法的次数
                        if avg_l1_distance > ref_l1_distance:
                            greater_data[key][method] += 1
            else:
                analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"
                if not os.path.exists(analyze1_filename):
                    continue

                analyze1_df = pd.read_csv(analyze1_filename, index_col=0)
                
                l1_distance = 0
                for i in range(1, 7):
                    param_diff = analyze1_df.loc[f"Param {i} Diff (Ground Truth)", "Sample_0"]
                    if pd.notna(param_diff):
                        l1_distance += abs(param_diff)
                
                # 计算比值
                if ref_l1_distance != 0:
                    ratio = l1_distance / ref_l1_distance
                    ratio_data[key][method].append(ratio)
                    # 统计大于参考方法的次数
                    if l1_distance > ref_l1_distance:
                        greater_data[key][method] += 1

# 创建最终的统计表格
# 1. 原始值统计 (_stats)
stats_df = pd.DataFrame(index=all_methods)
for key in stats_data:
    for method in all_methods:
        values = stats_data[key][method]
        if values:
            stats_df.loc[method, key] = np.mean(values)
        else:
            stats_df.loc[method, key] = np.nan

# 2. 比值统计 (_stats_ratio)
ratio_df = pd.DataFrame(index=comparison_methods)
for key in ratio_data:
    for method in comparison_methods:
        ratios = ratio_data[key][method]
        if ratios:
            ratio_df.loc[method, key] = np.mean(ratios)
        else:
            ratio_df.loc[method, key] = np.nan

# 3. 大于参考方法的比例统计 (_stats_greater)
greater_df = pd.DataFrame(index=comparison_methods)
for key in greater_data:
    total_count = len(stats_data[key][reference_method])  # 使用参考方法的样本数作为总数
    if total_count > 0:
        for method in comparison_methods:
            greater_df.loc[method, key] = greater_data[key][method] / total_count
    else:
        for method in comparison_methods:
            greater_df.loc[method, key] = np.nan

# 转置DataFrame
stats_df = stats_df.T
ratio_df = ratio_df.T
greater_df = greater_df.T

# 保存统计结果
stats_df.to_csv(f"{ANALYZE5_DIR}/exp3_stats.csv")
ratio_df.to_csv(f"{ANALYZE5_DIR}/exp3_stats_ratio.csv")
greater_df.to_csv(f"{ANALYZE5_DIR}/exp3_stats_greater.csv")

# 打印统计结果
print("\nOriginal Value Statistics:")
print(stats_df.to_string())
print("\nRatio Statistics (Ratio to bds-hilbert):")
print(ratio_df.to_string())
print("\nGreater Ratio Statistics:")
print(greater_df.to_string())
