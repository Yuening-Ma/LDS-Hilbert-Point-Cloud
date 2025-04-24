import os
import pandas as pd
import numpy as np
from scipy import stats
import itertools

# --- 配置 ---

M = 'int' # Data identifier for dim_3/5
SAMPLE_RATIO = 10 # Sample rate %
ALPHA = 0.05 # Significance level

# Directory settings for dim_3/5
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'
ANALYZE2_DIR = f'dim_3/5_closed_geometry_2/analyze2_M{M}/{SAMPLE_RATIO}/'
OUTPUT_DIR = f'dim_3/5_closed_geometry_2/paired_t_test/'

# Method definitions (confirm these match columns in _1st_bias.csv)
REFERENCE_METHOD = 'bds-hilbert'
random_seeds = [7, 42, 1309]
methods_srs = [f'srs_{seed}' for seed in random_seeds]
methods_fps = [f'fps_{seed}' for seed in random_seeds]
# Keep original comparison methods with seeds for loading
ORIGINAL_COMPARISON_METHODS_WITH_SEEDS = methods_srs + methods_fps + ['voxel', 'bds-pca']
ALL_METHODS_WITH_SEEDS = ORIGINAL_COMPARISON_METHODS_WITH_SEEDS + [REFERENCE_METHOD]

# Define the final comparison methods after averaging
FINAL_COMPARISON_METHODS = ['srs', 'fps', 'voxel', 'bds-pca']
FINAL_ALL_METHODS = FINAL_COMPARISON_METHODS + [REFERENCE_METHOD]

# 方法名称映射（用于输出表格）
method_name_mapping = {
    'srs': 'SRS',
    'fps': 'FPS',
    'bds-pca': 'LDS-PCA',
    'voxel': 'VOXEL',
    'bds-hilbert': 'LDS-HILBERT'
}

# Metrics to analyze (key: row name in CSV, value: concise name for output file)
METRICS_TO_ANALYZE = {
    'Hausdorff 95% Distance (Sample to GT)': 'H95_Sample_to_GT',
    'Hausdorff 95% Distance (GT to Sample)': 'H95_GT_to_Sample',
    'Chamfer Distance': 'Chamfer_Distance',
    'Avg Distance to Shape': 'p2m_Distance', # Matches 'Avg Distance to Shape' in CSV
    'VFH Distance': 'VFH_Distance'
}

# 创建N值分组，每15个ID一组
def create_id_groups(data_ids):
    N_groups = {}
    # 使用固定的N值
    n_values = [9180, 37240, 95760]
    
    # 按照每15个ID一组的顺序分组
    sorted_ids = sorted(data_ids)
    
    for i, n_value in enumerate(n_values):
        start_idx = i * 15
        end_idx = start_idx + 15
        if start_idx < len(sorted_ids):
            # 获取该组的ID列表，确保不超出数据范围
            if end_idx > len(sorted_ids):
                end_idx = len(sorted_ids)
            ids_for_n = sorted_ids[start_idx:end_idx]
            N_groups[n_value] = ids_for_n
    
    return N_groups, n_values

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---

# Read data info (get data_id list)
try:
    data_info = pd.read_csv(DATA_DIR + 'data_info.csv')
    data_ids = data_info['ID'].unique().tolist()
    # 创建基于N值的分组
    id_groups, n_values = create_id_groups(data_ids)
except Exception as e:
    print(f"Error reading data_info.csv: {e}")
    exit()

# Dictionary to hold DataFrames for each metric (loading ALL methods with seeds)
metric_data_raw = {metric_key: pd.DataFrame(index=data_ids, columns=ALL_METHODS_WITH_SEEDS, dtype=float)
                   for metric_key in METRICS_TO_ANALYZE.keys()}

print(f"Loading metric data from {ANALYZE2_DIR}...")

for data_id in data_ids:
    analyze2_file = f"{ANALYZE2_DIR}{data_id:04}_1st_bias.csv" # Use the correct filename
    if not os.path.exists(analyze2_file):
        print(f"  Warning: File not found {analyze2_file}, skipping data_id {data_id}")
        # Mark all metrics as NaN for this data_id
        for metric_key in METRICS_TO_ANALYZE.keys():
            metric_data_raw[metric_key].loc[data_id] = np.nan
        continue

    try:
        df_analyze2 = pd.read_csv(analyze2_file, index_col=0)

        for metric_key in METRICS_TO_ANALYZE.keys():
            if metric_key not in df_analyze2.index:
                print(f"  Warning: Metric '{metric_key}' not found in {analyze2_file}")
                metric_data_raw[metric_key].loc[data_id] = np.nan
                continue

            for method in ALL_METHODS_WITH_SEEDS: # Load all including seeds
                if method in df_analyze2.columns:
                    metric_value = df_analyze2.loc[metric_key, method]
                    metric_data_raw[metric_key].loc[data_id, method] = metric_value
                else:
                    # print(f"  Warning: Method '{method}' not found for metric '{metric_key}' in {analyze2_file}")
                    metric_data_raw[metric_key].loc[data_id, method] = np.nan # Mark as NaN if method column missing

    except Exception as e:
        print(f"  Error processing file {analyze2_file}: {e}")
        # Mark all metrics as NaN for this data_id if error occurs
        for metric_key in METRICS_TO_ANALYZE.keys():
            metric_data_raw[metric_key].loc[data_id] = np.nan

print("Finished loading raw data.")

# --- Average Seeded Methods and Prepare Final Data --- 
print("Averaging results for srs and fps methods...")
metric_data_final = {}

for metric_key, df_raw in metric_data_raw.items():
    df_final = pd.DataFrame(index=df_raw.index)
    # Calculate mean for srs if columns exist
    srs_cols_present = [col for col in methods_srs if col in df_raw.columns]
    if srs_cols_present:
        df_final['srs'] = df_raw[srs_cols_present].mean(axis=1)
    else:
        df_final['srs'] = np.nan
        
    # Calculate mean for fps if columns exist
    fps_cols_present = [col for col in methods_fps if col in df_raw.columns]
    if fps_cols_present:
        df_final['fps'] = df_raw[fps_cols_present].mean(axis=1)
    else:
        df_final['fps'] = np.nan
        
    # Copy other methods
    other_methods = ['voxel', 'bds-pca', REFERENCE_METHOD]
    for method in other_methods:
        if method in df_raw.columns:
            df_final[method] = df_raw[method]
        else:
             df_final[method] = np.nan
             
    metric_data_final[metric_key] = df_final
    
print("Finished averaging.")

# --- 创建多级索引的结果表 ---
# 使用映射后的方法名作为列名
mapped_method_names = [method_name_mapping[method] for method in FINAL_COMPARISON_METHODS]

# 创建多级索引的表格：一级索引是N值，二级索引是指标名
index_tuples = []
for n_value in n_values:
    for metric_key, metric_name in METRICS_TO_ANALYZE.items():
        index_tuples.append((n_value, metric_name))

# 创建MultiIndex
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['N', 'Metric'])

# 创建结果表
pvalue_table = pd.DataFrame(index=multi_index, columns=mapped_method_names)
tvalue_table = pd.DataFrame(index=multi_index, columns=mapped_method_names)

print("\n--- Performing paired t-tests by group and metric ---")

# 按N值分组分析
for n_value in n_values:
    print(f"\nProcessing group with N = {n_value}")
    group_ids = id_groups[n_value]
    
    # 对每个指标进行分析
    for metric_key, metric_name in METRICS_TO_ANALYZE.items():
        print(f"  Analyzing metric: {metric_name}")
        
        # 提取该组该指标的数据
        current_metric_df = metric_data_final[metric_key].loc[group_ids].copy()
        
        # 移除含NaN值的行
        original_rows = len(current_metric_df)
        current_metric_df.dropna(subset=FINAL_ALL_METHODS, how='any', inplace=True)
        removed_rows = original_rows - len(current_metric_df)
        if removed_rows > 0:
            print(f"    Removed {removed_rows} rows with NaN values.")
            
        if current_metric_df.empty:
            print(f"    No valid data for t-test for N={n_value}, metric={metric_name}")
            # 在结果表中标记为NaN
            for method in FINAL_COMPARISON_METHODS:
                mapped_name = method_name_mapping[method]
                pvalue_table.loc[(n_value, metric_name), mapped_name] = np.nan
                tvalue_table.loc[(n_value, metric_name), mapped_name] = np.nan
            continue
            
        print(f"    Performing tests across {len(current_metric_df)} valid data IDs...")
        reference_values = current_metric_df[REFERENCE_METHOD]
        
        # 对每个比较方法进行t检验
        for method in FINAL_COMPARISON_METHODS:
            mapped_name = method_name_mapping[method]
            comparison_values = current_metric_df[method]
            
            try:
                # 确保有足够的非NaN对进行t检验
                mask = ~np.isnan(comparison_values) & ~np.isnan(reference_values)
                comp_valid = comparison_values[mask]
                ref_valid = reference_values[mask]
                
                if len(comp_valid) < 2:  # 至少需要2对
                    raise ValueError(f"Not enough valid pairs ({len(comp_valid)}) for t-test")
                    
                # 执行配对t检验
                t_statistic, p_value = stats.ttest_rel(comp_valid, ref_valid)
                
                # 保存到表格
                pvalue_table.loc[(n_value, metric_name), mapped_name] = p_value
                tvalue_table.loc[(n_value, metric_name), mapped_name] = t_statistic
                
                print(f"      {mapped_name} vs {method_name_mapping[REFERENCE_METHOD]}: t={t_statistic:.4f}, p={p_value:.4f}")
                
            except Exception as e:
                print(f"      Error in t-test: {e}")
                pvalue_table.loc[(n_value, metric_name), mapped_name] = np.nan
                tvalue_table.loc[(n_value, metric_name), mapped_name] = np.nan

# --- 创建带显著性标记的t值表 ---
# 创建格式化的T值表，根据p值添加显著性标注
formatted_t_table = pd.DataFrame(index=multi_index, columns=mapped_method_names)

for idx in multi_index:
    n_value, metric_name = idx
    for method_name in mapped_method_names:
        t_value = tvalue_table.loc[idx, method_name]
        p_value = pvalue_table.loc[idx, method_name]
        
        if pd.isna(t_value) or pd.isna(p_value):
            formatted_t_table.loc[idx, method_name] = 'N/A'
        else:
            # 格式化t值为四位小数
            t_formatted = f"{t_value:.4f}"
            
            # 根据p值添加显著性标记，只有在t>0时才添加
            if t_value > 0:  # 只有当t>0时才添加显著性标记
                if p_value < 0.01:
                    t_formatted += "**"  # 非常显著 (p < 0.01)
                elif p_value < 0.05:
                    t_formatted += "*"   # 显著 (p < 0.05)
                
            formatted_t_table.loc[idx, method_name] = t_formatted

# --- 保存结果 --- 
p_values_file = f"{OUTPUT_DIR}p_values_by_N_metric_{SAMPLE_RATIO}percent.csv"
t_values_file = f"{OUTPUT_DIR}t_values_by_N_metric_{SAMPLE_RATIO}percent.csv"
formatted_t_file = f"{OUTPUT_DIR}t_values_with_significance_by_N_metric_{SAMPLE_RATIO}percent.csv"

# 保存P值表
pvalue_table.to_csv(p_values_file, float_format='%.6g')
print(f"\nP-values table saved to {p_values_file}")

# 保存T统计量表
tvalue_table.to_csv(t_values_file, float_format='%.6g')
print(f"T-statistics table saved to {t_values_file}")

# 保存带显著性标记的T值表
formatted_t_table.to_csv(formatted_t_file)
print(f"T-values table with significance marks saved to {formatted_t_file}")

print("\nPaired t-test script finished.")
