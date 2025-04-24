import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置数据目录和分析结果目录
M = 'int'
DATA_DIR = f'dim_3/4_surface_noised/data_M{M}/'  # 原始数据目录
ANALYZE1_DIR = f'dim_3/4_surface_noised/analyze1_M{M}/'  # 分析结果目录
OUTPUT_DIR = f'code_for_essay/output/exp2/'

# 定义分布类型和 N 值
surfaces = ['Plane', 'Sphere', 'Ellipsoid', 'Torus', 'Cylinder', 'Cone', 'Paraboloid']
n_values = [9180, 37240, 95760]  # 总点数
sample_ratios = [0.10, 0.20]  # 采样率
sample_methods = ['srs', 'fps', 'voxel', 'bds-pca', 'bds-hilbert']  # 降采样方法

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 创建一个 4 行 2 列的子图
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 18))
axes = axes.flatten()  # 展平 axes 以便按顺序使用

# 定义颜色和标记
colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'v', 'D']

# 遍历每个曲面
for i, surface in enumerate(surfaces):
    if i >= len(axes) - 1:  # 最后一个子图留空
        break

    # 初始化存储结果的字典
    results = {method: [] for method in sample_methods}

    # 遍历每个数据量和采样率
    for n in n_values:
        # 获取对应的数据 ID
        data_id = data_info[(data_info['Surface'] == surface) & (data_info['DataPoints'] == n)]['ID'].values[0]

        for sample_ratio in sample_ratios:
            label = f'{n}_{int(sample_ratio * 100)}%'  # 构造标签

            for method in sample_methods:
                # 从分析结果中获取估计的参数
                if method in ['srs', 'fps']:
                    l1_distances = []
                    for seed in [7, 42, 1309]:
                        analyze_file = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_{seed}_analyze1.csv"
                        if os.path.exists(analyze_file):
                            analyze_data = pd.read_csv(analyze_file, index_col=0)
                            diff_rows = [idx for idx in analyze_data.index if 'Param' in idx and 'Diff (Ground Truth)' in idx]
                            if 'Sample_0' in analyze_data.columns:
                                l1_distance = analyze_data.loc[diff_rows, 'Sample_0'].sum()
                            else:
                                l1_distance = analyze_data.loc[diff_rows].sum().values[0]
                            l1_distances.append(l1_distance)
                    if l1_distances:
                        l1_distance = np.mean(l1_distances)  # 计算三次随机数种子的均值
                        results[method].append((label, l1_distance))
                else:
                    analyze_file = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"
                    if os.path.exists(analyze_file):
                        analyze_data = pd.read_csv(analyze_file, index_col=0)
                        diff_rows = [idx for idx in analyze_data.index if 'Param' in idx and 'Diff (Ground Truth)' in idx]
                        if 'Sample_0' in analyze_data.columns:
                            l1_distance = analyze_data.loc[diff_rows, 'Sample_0'].sum()
                        else:
                            l1_distance = analyze_data.loc[diff_rows].sum().values[0]
                        results[method].append((label, l1_distance))


    # 绘制折线图
    for j, method in enumerate(sample_methods):
        x_labels = [item[0] for item in results[method]]
        y_values = [item[1] for item in results[method]]
        axes[i].plot(x_labels, y_values, label=method.upper(), color=colors[j], marker=markers[j], linestyle='-')

    # 设置子图标题和标签
    axes[i].set_title(surface, fontsize=12)
    axes[i].set_xlabel('N and Sample Ratio', fontsize=10)
    axes[i].set_ylabel('L1 Distance', fontsize=10)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)  # 倾斜 x 轴标签
    axes[i].legend(loc='upper right', fontsize=8)
    axes[i].set_yscale('function', functions=(np.sqrt, np.square))

# 最后一个子图为空
axes[-1].axis('off')

# 调整子图间距
plt.tight_layout(pad=1.0)  # 将四边距设置为 1.0
plt.subplots_adjust(hspace=0.5, wspace=0.2)  # 调整行间距和列间距

# 保存和显示图像
plt.savefig(f'{OUTPUT_DIR}exp2_dim3_L1.svg', format='svg')
plt.savefig(f'{OUTPUT_DIR}exp2_dim3_L1.png', dpi=300)
plt.show()