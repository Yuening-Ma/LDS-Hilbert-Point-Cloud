import os
import pandas as pd
import numpy as np

# 设置数据目录
M = 'int'
DATA_DIR = f'dim_3/4_surface_noised/data_M{M}/'  # 原始数据目录
ANALYZE1_DIR = f'dim_3/4_surface_noised/analyze1_M{M}/'  # 分析结果目录
OUTPUT_DIR = f'code_for_essay/output/exp2/'

# 定义曲面类型和 N 值
surfaces = ['Plane', 'Sphere', 'Ellipsoid', 'Torus', 'Cylinder', 'Cone', 'Paraboloid']
n_values = [9180, 37240, 95760]  # 总点数
sample_ratios = [0.10, 0.20]  # 采样率
methods = ['srs', 'fps', 'voxel', 'bds-pca', 'bds-hilbert']  # 降采样方法
random_seeds = [7, 42, 1309]

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 初始化结果字典
results = {method: {} for method in methods}

# 遍历每个曲面
for surface in surfaces:
    # 获取当前曲面的所有数据
    surface_data = data_info[data_info['Surface'] == surface]

    # 遍历每个数据点
    for _, row in surface_data.iterrows():
        data_id = row['ID']
        n = row['DataPoints']  # 获取 N 的值

        # 遍历每个采样率
        for sample_ratio in sample_ratios:
            label = f'{n}_{int(sample_ratio * 100)}%'  # 构造标签

            # 处理 SRS 和 FPS（需要取三个随机种子的均值）
            for method in ['srs', 'fps']:
                l1_distances = []
                for seed in random_seeds:
                    filename = f'{ANALYZE1_DIR}{int(sample_ratio * 100)}/{data_id:04}_{method}_{seed}_analyze1.csv'
                    if os.path.exists(filename):
                        df = pd.read_csv(filename, header=0, index_col=0)
                        # 计算L1距离：所有参数diff值的和
                        diff_rows = [idx for idx in df.index if 'Param' in idx and 'Diff (Ground Truth)' in idx]
                        if 'Sample_0' in df.columns:
                            l1_distance = df.loc[diff_rows, 'Sample_0'].sum()
                        else:
                            l1_distance = df.loc[diff_rows].sum().values[0]
                        l1_distances.append(float(l1_distance))
                if l1_distances:
                    if label not in results[method]:
                        results[method][label] = []
                    results[method][label].append(np.mean(l1_distances))

            # 处理 BDS, Voxel
            for method in ['voxel', 'bds-pca', 'bds-hilbert']:
                filename = f'{ANALYZE1_DIR}{int(sample_ratio * 100)}/{data_id:04}_{method}_analyze1.csv'
                if os.path.exists(filename):
                    df = pd.read_csv(filename, header=0, index_col=0)
                    # 计算L1距离：所有参数diff值的和
                    diff_rows = [idx for idx in df.index if 'Param' in idx and 'Diff (Ground Truth)' in idx]
                    if 'Sample_0' in df.columns:
                        l1_distance = df.loc[diff_rows, 'Sample_0'].sum()
                    else:
                        l1_distance = df.loc[diff_rows].sum().values[0]
                    if label not in results[method]:
                        results[method][label] = []
                    results[method][label].append(float(l1_distance))

# 计算每个标签的平均值
mean_results = {}
for method in methods:
    mean_results[method] = {label: np.mean(values) for label, values in results[method].items()}

# 创建DataFrame
df = pd.DataFrame(mean_results)

# 保存为CSV文件
df.to_csv(f'{OUTPUT_DIR}dim3_l1_mean.csv')

# 打印结果
print("\n平均L1距离结果：")
print(df) 