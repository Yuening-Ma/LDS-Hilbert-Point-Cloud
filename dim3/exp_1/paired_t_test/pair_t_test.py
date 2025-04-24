import os
import pandas as pd
import numpy as np
from scipy import stats
import itertools # Import itertools

# --- 配置 --- 

M = 'Mint' # 数据标识
SAMPLE_RATIOS = [0.1, 0.2] # 采样率 - 统计两种采样率
ALPHA = 0.05 # 显著性水平

# 目录设置
DATA_DIR = f'dim_3/4_surface_noised/data_{M}/'
OUTPUT_DIR = f'dim_3/4_surface_noised/paired_t_test/'

# 方法定义
REFERENCE_METHOD = 'bds-hilbert'
random_seeds = [7, 42, 1309] # Define random seeds used
methods_srs = [f'srs_{seed}' for seed in random_seeds]
methods_fps = [f'fps_{seed}' for seed in random_seeds]
# Keep original comparison methods with seeds for loading
ORIGINAL_COMPARISON_METHODS_WITH_SEEDS = methods_srs + methods_fps + ['bds-pca', 'voxel'] 
ALL_METHODS_WITH_SEEDS = ORIGINAL_COMPARISON_METHODS_WITH_SEEDS + [REFERENCE_METHOD]

# Define the final comparison methods after averaging
FINAL_COMPARISON_METHODS = ['srs', 'fps', 'bds-pca', 'voxel']
FINAL_ALL_METHODS = FINAL_COMPARISON_METHODS + [REFERENCE_METHOD]

# 方法名称映射（用于输出表格）
method_name_mapping = {
    'srs': 'SRS',
    'fps': 'FPS',
    'bds-pca': 'LDS-PCA',
    'voxel': 'VOXEL',
    'bds-hilbert': 'LDS-HILBERT',
}

# 点数分组定义（每组7个对象，共3组，对应不同点数）
n_values = [9180, 37240, 95760]  # 每组的点数
data_id_groups = {}
# 生成data_id分组
for i, n in enumerate(n_values):
    start_id = i * 7
    end_id = start_id + 7
    group_ids = list(range(start_id, end_id))
    data_id_groups[n] = group_ids

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用映射后的方法名作为列名
mapped_method_names = [method_name_mapping[method] for method in FINAL_COMPARISON_METHODS]

# 创建多级索引：一级是N值，二级是采样率
index_tuples = []
for n_value in n_values:
    for sample_ratio in SAMPLE_RATIOS:
        index_tuples.append((n_value, sample_ratio))

# 创建MultiIndex
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['N', 'SAMPLE_RATIO'])

# 创建带显著性标记的t值表
formatted_t_table = pd.DataFrame(index=multi_index, columns=mapped_method_names)

# 对每个采样率进行处理
for SAMPLE_RATIO in SAMPLE_RATIOS:
    print(f"\n--- Processing SAMPLE_RATIO: {SAMPLE_RATIO} ---")
    # 文件夹名称使用百分比整数值
    FOLDER_RATIO = int(SAMPLE_RATIO * 100)
    ANALYZE2_DIR = f'dim_3/4_surface_noised/analyze2_{M}/{FOLDER_RATIO}/'
    
    # --- 数据加载与准备 --- 
    # 读取数据信息
    try:
        data_info = pd.read_csv(DATA_DIR + 'data_info.csv')
        data_ids = list(range(0, 21))  # 直接使用0～20的data_id
    except Exception as e:
        print(f"Error reading data_info.csv from {DATA_DIR}: {e}")
        continue

    # 存储所有方法（包括种子）对所有data_id的L1距离
    l1_distances_raw = pd.DataFrame(index=data_ids, columns=ALL_METHODS_WITH_SEEDS, dtype=float)

    print(f"Loading L1 distances from {ANALYZE2_DIR}...")

    for data_id in data_ids:
        # 使用正确的文件名格式
        analyze2_file = f"{ANALYZE2_DIR}{data_id:04}_all_biases.csv" 
        if not os.path.exists(analyze2_file):
            print(f"  Warning: File not found {analyze2_file}, skipping data_id {data_id}")
            l1_distances_raw.loc[data_id] = np.nan # Mark row as NaN
            continue
            
        try:
            df_analyze2 = pd.read_csv(analyze2_file, index_col=0)
            # 找到所有参数差异的行 (使用均值行)
            param_diff_rows = [idx for idx in df_analyze2.index if idx.startswith('Param') and idx.endswith('Diff (Ground Truth) Mean')]
            
            if not param_diff_rows:
                print(f"  Warning: No 'Param X Diff (Ground Truth) Mean' rows found in {analyze2_file} for ID {data_id}")
                l1_distances_raw.loc[data_id] = np.nan # Mark row as NaN
                continue

            for method in ALL_METHODS_WITH_SEEDS: # Load all including seeds
                if method in df_analyze2.columns:
                    # 提取对应方法的参数差异值并求和得到L1距离
                    l1_dist = df_analyze2.loc[param_diff_rows, method].sum()
                    l1_distances_raw.loc[data_id, method] = l1_dist
                else:
                    l1_distances_raw.loc[data_id, method] = np.nan # 标记为NaN如果方法不存在
                    
        except Exception as e:
            print(f"  Error processing file {analyze2_file}: {e}")
            l1_distances_raw.loc[data_id] = np.nan # 出错则整行标记为NaN

    print("Finished loading raw data.")

    # --- Average Seeded Methods and Prepare Final Data --- 
    print("Averaging results for srs and fps methods...")
    l1_distances_final = pd.DataFrame(index=l1_distances_raw.index)

    # Calculate mean for srs if columns exist
    srs_cols_present = [col for col in methods_srs if col in l1_distances_raw.columns]
    if srs_cols_present:
        l1_distances_final['srs'] = l1_distances_raw[srs_cols_present].mean(axis=1)
    else:
        l1_distances_final['srs'] = np.nan
        
    # Calculate mean for fps if columns exist
    fps_cols_present = [col for col in methods_fps if col in l1_distances_raw.columns]
    if fps_cols_present:
        l1_distances_final['fps'] = l1_distances_raw[fps_cols_present].mean(axis=1)
    else:
        l1_distances_final['fps'] = np.nan
        
    # Copy other methods
    other_methods = ['voxel', 'bds-pca', REFERENCE_METHOD]
    for method in other_methods:
        if method in l1_distances_raw.columns:
            l1_distances_final[method] = l1_distances_raw[method]
        else:
            l1_distances_final[method] = np.nan
             
    print("Finished averaging.")

    # --- 执行配对 T 检验 (按点数分组) --- 
    print("Performing paired t-tests by point count groups...")
    
    # 创建临时存储t值和p值的表
    tmp_tvalue_table = pd.DataFrame(index=n_values, columns=FINAL_COMPARISON_METHODS)
    tmp_pvalue_table = pd.DataFrame(index=n_values, columns=FINAL_COMPARISON_METHODS)

    # 按点数进行分组分析
    for n_value, group_ids in data_id_groups.items():
        print(f"\n--- Processing group with point count: {n_value} ---")
        # 提取该组的数据
        group_data = l1_distances_final.loc[group_ids]
        
        # 移除含有NaN的行
        group_data_clean = group_data.dropna(subset=FINAL_ALL_METHODS, how='any')
        
        if group_data_clean.empty:
            print(f"  No valid data for point count {n_value} after removing NaNs.")
            # 在表中标记为空
            for method in FINAL_COMPARISON_METHODS:
                tmp_tvalue_table.loc[n_value, method] = np.nan
                tmp_pvalue_table.loc[n_value, method] = np.nan
            continue
        
        print(f"  Performing tests across {len(group_data_clean)} valid data IDs...")
        reference_values = group_data_clean[REFERENCE_METHOD]
        
        # 对每个比较方法进行t检验
        for method in FINAL_COMPARISON_METHODS:
            comparison_values = group_data_clean[method]
            
            try:
                # 确保有足够的数据对进行t检验
                mask = ~np.isnan(comparison_values) & ~np.isnan(reference_values)
                comp_valid = comparison_values[mask]
                ref_valid = reference_values[mask]
                
                if len(comp_valid) < 2:  # 至少需要2对数据
                    raise ValueError(f"Not enough valid pairs ({len(comp_valid)}) for t-test")
                    
                t_statistic, p_value = stats.ttest_rel(comp_valid, ref_valid)
                
                # 记录到临时表中
                tmp_tvalue_table.loc[n_value, method] = t_statistic
                tmp_pvalue_table.loc[n_value, method] = p_value
                
                print(f"    {method_name_mapping[method]} vs {method_name_mapping[REFERENCE_METHOD]}: t={t_statistic:.4f}, p={p_value:.4f}")
                
            except Exception as e:
                print(f"    Error in t-test for {method} with point count {n_value}: {e}")
                tmp_tvalue_table.loc[n_value, method] = np.nan
                tmp_pvalue_table.loc[n_value, method] = np.nan

    # 将临时结果整合到带显著性标记的t值表中
    for n_value in n_values:
        for method in FINAL_COMPARISON_METHODS:
            mapped_name = method_name_mapping[method]
            t_value = tmp_tvalue_table.loc[n_value, method]
            p_value = tmp_pvalue_table.loc[n_value, method]
            
            if pd.isna(t_value) or pd.isna(p_value):
                formatted_t_table.loc[(n_value, SAMPLE_RATIO), mapped_name] = 'N/A'
            else:
                # 格式化t值为四位小数
                t_formatted = f"{t_value:.4f}"
                
                # 根据p值添加显著性标记，只有在t>0时才添加
                if t_value > 0:  # 只有当t>0时才添加显著性标记
                    if p_value < 0.01:
                        t_formatted += "**"  # 非常显著 (p < 0.01)
                    elif p_value < 0.05:
                        t_formatted += "*"   # 显著 (p < 0.05)
                    
                formatted_t_table.loc[(n_value, SAMPLE_RATIO), mapped_name] = t_formatted

# --- 保存结果 --- 
# 保存带显著性标记的t值表
formatted_t_file = f"{OUTPUT_DIR}t_values_with_significance_by_N_and_SAMPLE_RATIO.csv"
formatted_t_table.to_csv(formatted_t_file)
print(f"\nT-values table with significance marks saved to {formatted_t_file}")

print("\nPaired t-test script finished.")
