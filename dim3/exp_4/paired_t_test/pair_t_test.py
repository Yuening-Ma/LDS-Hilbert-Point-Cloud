import os
import pandas as pd
import numpy as np
from scipy import stats
import itertools

# --- 配置 ---

M = 'flexible' # 根据实验文件设置为'flexible'
SAMPLE_RATIO = 10 # 采样率 % - Stanford_all实验只使用了10%的采样率
ALPHA = 0.05 # 显著性水平

# 目录设置，根据7_stanford_all实验结构
DATA_DIR = f'dim_3/7_stanford_all/Stanford_all/'
ANALYZE2_DIR = f'dim_3/7_stanford_all/analyze2_M{M}/{SAMPLE_RATIO}/'
OUTPUT_DIR = f'dim_3/7_stanford_all/paired_t_test/'

# 方法定义（确认与_1st_bias.csv列匹配）
REFERENCE_METHOD = 'bds-hilbert'
random_seeds = [7, 42, 1309]
methods_srs = [f'srs_{seed}' for seed in random_seeds]
methods_fps = [f'fps_{seed}' for seed in random_seeds]
# 保留原始带种子的比较方法用于加载
ORIGINAL_COMPARISON_METHODS_WITH_SEEDS = methods_srs + methods_fps + ['voxel', 'bds-pca']
ALL_METHODS_WITH_SEEDS = ORIGINAL_COMPARISON_METHODS_WITH_SEEDS + [REFERENCE_METHOD]

# 定义平均后的最终比较方法
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

# 要分析的指标（键：CSV中的行名，值：用于输出文件的简洁名称）
METRICS_TO_ANALYZE = {
    'Hausdorff 95% Distance (Sample to GT)': 'H95_Sample_to_GT',
    'Hausdorff 95% Distance (GT to Sample)': 'H95_GT_to_Sample',
    'Chamfer Distance': 'Chamfer_Distance',
    'Avg Distance to Shape': 'p2m_Distance', # 匹配CSV中的'Avg Distance to Shape'
    'VFH Distance': 'VFH_Distance',
}

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 数据加载 ---

# 从pc目录获取数据ID列表
try:
    data_ids = set()
    for file_name in os.listdir(DATA_DIR + 'pc/'):
        if file_name.endswith('.csv'):
            data_id = file_name.split('.')[0]
            data_ids.add(data_id)
    data_ids = sorted(list(data_ids))
except Exception as e:
    print(f"Error reading data directory: {e}")
    exit()

# 为每个指标创建保存DataFrame的字典（加载所有带种子的方法）
metric_data_raw = {metric_key: pd.DataFrame(index=data_ids, columns=ALL_METHODS_WITH_SEEDS, dtype=float)
                   for metric_key in METRICS_TO_ANALYZE.keys()}

print(f"Loading metric data from {ANALYZE2_DIR}...")

for data_id in data_ids:
    analyze2_file = f"{ANALYZE2_DIR}{data_id}_1st_bias.csv" # 使用正确的文件名格式
    if not os.path.exists(analyze2_file):
        print(f"  Warning: File not found {analyze2_file}, skipping data_id {data_id}")
        # 将该data_id的所有指标标记为NaN
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

            for method in ALL_METHODS_WITH_SEEDS: # 加载所有包括种子的方法
                if method in df_analyze2.columns:
                    metric_value = df_analyze2.loc[metric_key, method]
                    metric_data_raw[metric_key].loc[data_id, method] = metric_value
                else:
                    # print(f"  Warning: Method '{method}' not found for metric '{metric_key}' in {analyze2_file}")
                    metric_data_raw[metric_key].loc[data_id, method] = np.nan # 如果方法列缺失则标记为NaN

    except Exception as e:
        print(f"  Error processing file {analyze2_file}: {e}")
        # 如果发生错误，将该data_id的所有指标标记为NaN
        for metric_key in METRICS_TO_ANALYZE.keys():
            metric_data_raw[metric_key].loc[data_id] = np.nan

print("Finished loading raw data.")

# --- 平均有种子的方法并准备最终数据 --- 
print("Averaging results for srs and fps methods...")
metric_data_final = {}

for metric_key, df_raw in metric_data_raw.items():
    df_final = pd.DataFrame(index=df_raw.index)
    # 如果列存在，计算srs的平均值
    srs_cols_present = [col for col in methods_srs if col in df_raw.columns]
    if srs_cols_present:
        df_final['srs'] = df_raw[srs_cols_present].mean(axis=1)
    else:
        df_final['srs'] = np.nan
        
    # 如果列存在，计算fps的平均值
    fps_cols_present = [col for col in methods_fps if col in df_raw.columns]
    if fps_cols_present:
        df_final['fps'] = df_raw[fps_cols_present].mean(axis=1)
    else:
        df_final['fps'] = np.nan
        
    # 复制其他方法
    other_methods = ['voxel', 'bds-pca', REFERENCE_METHOD]
    for method in other_methods:
        if method in df_raw.columns:
            df_final[method] = df_raw[method]
        else:
             df_final[method] = np.nan
             
    metric_data_final[metric_key] = df_final
    
print("Finished averaging.")

# --- 创建结果表 ---
# 使用映射后的方法名作为列索引
mapped_method_names = [method_name_mapping[method] for method in FINAL_COMPARISON_METHODS]

# 使用指标简洁名称作为行索引
metric_names = list(METRICS_TO_ANALYZE.values())

# 创建结果表
pvalue_table = pd.DataFrame(index=metric_names, columns=mapped_method_names)
tvalue_table = pd.DataFrame(index=metric_names, columns=mapped_method_names)
significant_table = pd.DataFrame(index=metric_names, columns=mapped_method_names)

print("\n--- Performing paired t-tests for all metrics ---")

# 对每个指标进行分析
for metric_key, metric_name in METRICS_TO_ANALYZE.items():
    print(f"\nAnalyzing metric: {metric_name}")
    
    # 获取该指标的数据
    current_metric_df = metric_data_final[metric_key].copy()
    
    # 删除包含任何FINAL方法NaN值的行
    original_rows = len(current_metric_df)
    current_metric_df.dropna(subset=FINAL_ALL_METHODS, how='any', inplace=True)
    removed_rows = original_rows - len(current_metric_df)
    if removed_rows > 0:
        print(f"  Removed {removed_rows} rows with NaN values.")
        
    if current_metric_df.empty:
        print(f"  No valid data for t-test for metric {metric_name}")
        # 在结果表中标记为NaN
        for method in FINAL_COMPARISON_METHODS:
            mapped_name = method_name_mapping[method]
            pvalue_table.loc[metric_name, mapped_name] = np.nan
            tvalue_table.loc[metric_name, mapped_name] = np.nan
            significant_table.loc[metric_name, mapped_name] = "N/A"
        continue
        
    print(f"  Performing tests across {len(current_metric_df)} valid data IDs...")
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
            significant = p_value < ALPHA
            
            # 保存到表格
            pvalue_table.loc[metric_name, mapped_name] = p_value
            tvalue_table.loc[metric_name, mapped_name] = t_statistic
            significant_table.loc[metric_name, mapped_name] = "是" if significant else "否"
            
            print(f"    {mapped_name} vs {method_name_mapping[REFERENCE_METHOD]}: t={t_statistic:.4f}, p={p_value:.4f}, Significant={significant}")
            
        except Exception as e:
            print(f"    Error in t-test for {mapped_name}: {e}")
            pvalue_table.loc[metric_name, mapped_name] = np.nan
            tvalue_table.loc[metric_name, mapped_name] = np.nan
            significant_table.loc[metric_name, mapped_name] = "Error"

# --- 创建带显著性标记的t值表 ---
# 创建格式化的T值表，根据p值添加显著性标注
formatted_t_table = pd.DataFrame(index=metric_names, columns=mapped_method_names)

for metric_name in metric_names:
    for method_name in mapped_method_names:
        t_value = tvalue_table.loc[metric_name, method_name]
        p_value = pvalue_table.loc[metric_name, method_name]
        
        if pd.isna(t_value) or pd.isna(p_value):
            formatted_t_table.loc[metric_name, method_name] = 'N/A'
        else:
            # 格式化t值为四位小数
            t_formatted = f"{t_value:.4f}"
            
            # 根据p值添加显著性标记，只有在t>0时才添加
            if t_value > 0:  # 只有当t>0时才添加显著性标记
                if p_value < 0.01:
                    t_formatted += "**"  # 非常显著 (p < 0.01)
                elif p_value < 0.05:
                    t_formatted += "*"   # 显著 (p < 0.05)
                
            formatted_t_table.loc[metric_name, method_name] = t_formatted

# --- 保存结果 --- 
p_values_file = f"{OUTPUT_DIR}p_values_by_metrics_{SAMPLE_RATIO}percent.csv"
t_values_file = f"{OUTPUT_DIR}t_values_by_metrics_{SAMPLE_RATIO}percent.csv"
formatted_t_file = f"{OUTPUT_DIR}t_values_with_significance_by_metrics_{SAMPLE_RATIO}percent.csv"

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
