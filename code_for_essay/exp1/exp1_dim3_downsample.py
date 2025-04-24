import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 设置数据目录和分析结果目录
M = 'int'
DATA_DIR = f'dim_3/4_surface_noised/data_M{M}/'  # 原始数据目录
SAMPLE_OUTPUT_DIR = f'dim_3/4_surface_noised/sample_M{M}/'  # 降采样结果目录
ANALYZE1_DIR = f'dim_3/4_surface_noised/analyze1_M{M}/'  # 分析结果目录
OUTPUT_DIR = f'code_for_essay/output/exp2/'

# 定义分布类型和 N 值
surfaces = ['Plane', 'Sphere']
n_value = 9180  # 选择 N=340 的数据
sample_ratio = 0.10  # 采样率
sample_methods = ['srs', 'fps', 'voxel', 'bds-pca', 'bds-hilbert']  # 降采样方法

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')
dist_data = data_info[data_info['DataPoints'] == n_value]

# 创建两个 fig 分别展示 Plane 和 Sphere 分布
for surface in surfaces:
    # 获取对应的数据 ID
    data_id = dist_data[dist_data['Surface'] == surface]['ID'].values[0]

    # 读取真值点云
    pc_ground_truth = np.loadtxt(f"{DATA_DIR}{data_id:04}.csv", delimiter=',')

    # 创建一个 2 行 3 列的子图
    fig = plt.figure(figsize=(14, 8))
    axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

    # 绘制真值曲面
    if surface == "Plane":
        a, b, c = map(float, dist_data[dist_data['Surface'] == surface]['Parameters'].values[0].split('_'))
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        x, y = np.meshgrid(x, y)
        z = a * x + b * y + c
    elif surface == "Sphere":
        x_center, y_center, z_center, radius = map(float, dist_data[dist_data['Surface'] == surface]['Parameters'].values[0].split('_'))
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = x_center + radius * np.sin(phi) * np.cos(theta)
        y = y_center + radius * np.sin(phi) * np.sin(theta)
        z = z_center + radius * np.cos(phi)

    axes[0].plot_surface(x, y, z, color='blue', alpha=0.3)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].set_xlabel('X', labelpad=10)
    axes[0].set_ylabel('Y', labelpad=10)
    axes[0].set_zlabel('Z', labelpad=10)

    # 绘制降采样结果和重建的曲面
    for i, method in enumerate(sample_methods, start=1):
        if method in ['srs', 'fps']:
            sample_file = f"{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_{method}_7.csv"
        else:
            sample_file = f"{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_{method}.csv"

        if os.path.exists(sample_file):
            sampled_data = np.loadtxt(sample_file, delimiter=',', skiprows=1)  # 指定分隔符为逗号

            # 如果有多组降采样结果，取第一组
            if sampled_data.ndim > 2:
                sampled_data = sampled_data[:, :, 0]

            # 绘制降采样点云
            axes[i].scatter(sampled_data[:, 0], sampled_data[:, 1], sampled_data[:, 2], s=3, color='red', alpha=1.0)

            # 从分析结果中获取估计的参数
            if method in ['srs', 'fps']:
                analyze_file = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_7_analyze1.csv"
            else:
                analyze_file = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"

            if os.path.exists(analyze_file):
                analyze_data = pd.read_csv(analyze_file, index_col=0)
                estimated_params = analyze_data.loc['Estimated Param 1':'Estimated Param 6', 'Sample_0'].values

                # 生成重建的曲面
                if surface == "Plane":
                    a, b, c = estimated_params[:3]
                    x = np.linspace(-2, 2, 100)
                    y = np.linspace(-2, 2, 100)
                    x, y = np.meshgrid(x, y)
                    z = a * x + b * y + c
                elif surface == "Sphere":
                    x_center, y_center, z_center, radius = estimated_params[:4]
                    theta = np.linspace(0, 2 * np.pi, 100)
                    phi = np.linspace(0, np.pi, 100)
                    theta, phi = np.meshgrid(theta, phi)
                    x = x_center + radius * np.sin(phi) * np.cos(theta)
                    y = y_center + radius * np.sin(phi) * np.sin(theta)
                    z = z_center + radius * np.cos(phi)

                # 绘制重建的曲面
                axes[i].plot_surface(x, y, z, color='blue', alpha=0.3)

            # 设置标题
            axes[i].set_title(f'{method.upper()}', fontsize=12)
            axes[i].set_xlabel('X', labelpad=10)
            axes[i].set_ylabel('Y', labelpad=10)
            axes[i].set_zlabel('Z', labelpad=10)

        else:
            print(f"Sample file not found for {method}.")

    # 调整子图间距
    plt.tight_layout(pad=0.5)  # 减小四边距
    plt.subplots_adjust(hspace=0.15, wspace=0.15, left=0.02, right=0.96, top=0.95, bottom=0.05)

    # 保存和显示图像
    plt.savefig(f'{OUTPUT_DIR}{surface}_downsampling_N{n_value}.svg', format='svg')
    plt.savefig(f'{OUTPUT_DIR}{surface}_downsampling_N{n_value}.png', dpi=300)
    plt.show()