import os
import pandas as pd
import numpy as np
import itertools

M = 'flexible'
DATA_DIR = f'dim_3/7_stanford_all/Stanford_all/'
PC_DIR = f"{DATA_DIR}pc/"
ANALYZE1_DIR = f'dim_3/7_stanford_all/analyze1_M{M}/'
ANALYZE2_DIR = f'dim_3/7_stanford_all/analyze2_M{M}/'
os.makedirs(ANALYZE2_DIR, exist_ok=True)

# 随机种子列表
random_seeds = [7, 42, 1309]

# 需要随机种子的方法
methods_need_seed = ['srs', 'fps']
# 生成方法和种子的组合
methods_random_combinations = [f"{method}_{seed}" for method, seed in itertools.product(methods_need_seed, random_seeds)]

# 不需要随机种子的方法
methods_no_seed = ['bds-pca', 'bds-hilbert', 'voxel']

# 所有方法列表
methods_all = methods_random_combinations + methods_no_seed

# 只处理10%采样比例
sample_ratio = 0.10
ratio_dir = f"{int(sample_ratio * 100)}"

# 直接处理采样比例目录
ratio_folder_path = os.path.join(ANALYZE1_DIR, ratio_dir)
if not os.path.isdir(ratio_folder_path):
    print(f"Ratio folder not found: {ratio_folder_path}")
    exit(1)

# 创建对应的分析结果输出目录
output_folder = os.path.join(ANALYZE2_DIR, ratio_dir)
os.makedirs(output_folder, exist_ok=True)
print(f"Processing ratio: {ratio_dir}")

# 从原始点云文件获取所有模型名称
model_names = [os.path.splitext(f)[0] for f in os.listdir(PC_DIR) if f.endswith('.csv')]
model_names.sort()  # 按字母顺序排序
print(f"Found {len(model_names)} models: {model_names}")

# 定义指标列表
metrics = [
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

# 为每个模型处理所有方法的分析结果
for model_name in model_names:
    print(f"\tProcessing model: {model_name}")
    
    # 创建一个DataFrame用于存储所有方法的第一个采样点云的指标
    all_first_samples_df = pd.DataFrame(index=metrics)
    
    # 处理所有方法（包括带随机种子和不带随机种子的方法）
    for method in methods_all:
        # 构建文件路径
        if '_' in method:  # 带随机种子的方法
            file_path = os.path.join(ratio_folder_path, f"{model_name}_{method}_analyze1.csv")
        else:  # 不带随机种子的方法
            file_path = os.path.join(ratio_folder_path, f"{model_name}_{method}_analyze1.csv")
        
        if not os.path.exists(file_path):
            print(f"\t\tFile not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            
            # 提取第一个采样点云的指标
            if 'Sample_0' in df.columns:
                # 提取所有指标值
                metrics_values = []
                for metric in metrics:
                    if metric in df.index:
                        metrics_values.append(df.loc[metric, 'Sample_0'])
                    else:
                        print(f"\t\tMissing metric {metric} in {file_path}")
                        metrics_values.append(np.nan)
                
                # 将结果添加到汇总DataFrame
                all_first_samples_df[method] = metrics_values
            else:
                print(f"\t\tNo Sample_0 column in {file_path}")
        except Exception as e:
            print(f"\t\tError processing {file_path}: {str(e)}")
    
    # 保存合并的结果
    if not all_first_samples_df.empty:
        output_path = os.path.join(output_folder, f"{model_name}_1st_bias.csv")
        all_first_samples_df.to_csv(output_path, float_format="%.10f")
        print(f"\t\tSaved: {output_path}")
