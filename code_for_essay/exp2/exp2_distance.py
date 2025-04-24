import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置分析结果目录
ANALYZE1_DIR = 'dim_3/5_closed_geometry_2/analyze1_Mint/'  # 分析结果目录
DATA_DIR = 'dim_3/5_closed_geometry_2/data_Mint/'  # 原始数据目录
OUTPUT_DIR = 'code_for_essay/output/exp3/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 定义几何体类型
geometries = [
    "Sphere", "Cylinder", "EquilateralTriangularPrism", "Cube",
    "QuadrangularPrism", "PentagonalPrism", "HexagonalPrism", "LBeam",
    "TBeam", "UBeam", "HBeam", "Cone", "Torus", "Tetrahedron", "Octahedron"
]

# 定义采样方法
sample_methods = ['srs', 'fps', 'voxel', 'bds-pca', 'bds-hilbert']

# 随机种子列表
random_seeds = [7, 42, 1309]

# 定义颜色和标记
colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'v', 'D']

# 数据点数列表
n_values = [9180, 37240, 95760]
# 生成等距的x坐标（1, 2, 3）对应三个N值
x_positions = [1, 2, 3]
sample_ratio = 10  # 10%

# 定义要绘制的平滑指标
metrics = {
    'Hausdorff 95% Distance (Sample to GT)': 'HD95_sample_to_gt',
    'Hausdorff 95% Distance (GT to Sample)': 'HD95_gt_to_sample',
    'Chamfer Distance': 'chamfer_distance',
    'VFH Distance': 'vfh_distance',
    'Avg Distance to Shape': 'pc_to_mesh_distance'
}

# 为每个指标初始化结果字典
metrics_results = {metric_name: {method: {} for method in sample_methods} for metric_name in metrics.keys()}

# 收集所有数据
for metric_name in metrics.keys():
    # 遍历每个几何体
    for geometry in geometries:
        # 获取对应的数据 ID
        geometry_data = data_info[data_info['Geometry'] == geometry]
        
        if geometry_data.empty:
            print(f"No data found for geometry: {geometry}")
            continue
            
        # 遍历所有N值
        for n in n_values:
            # 获取特定N值对应的数据行
            filtered_data = geometry_data[geometry_data['SampleNum'] == n]
            
            if filtered_data.empty:
                print(f"No data found for geometry {geometry} with N={n}")
                continue
                
            data_id = filtered_data['ID'].values[0]
            
            # 遍历每个采样方法
            for method in sample_methods:
                if method in ['srs', 'fps']:
                    # 对于 srs 和 fps，取三个随机数种子的均值
                    distances = []
                    for seed in random_seeds:
                        analyze_file = f"{ANALYZE1_DIR}{sample_ratio}/{data_id:04}_{method}_{seed}_analyze1.csv"
                        if os.path.exists(analyze_file):
                            analyze_data = pd.read_csv(analyze_file, index_col=0)
                            distance = analyze_data.loc[metric_name, 'Sample_0']
                            distances.append(distance)
                    
                    # 只有当至少有一个有效距离时才计算均值
                    if distances and not all(np.isnan(distances)):
                        # 过滤掉NaN值再计算均值
                        valid_distances = [d for d in distances if not np.isnan(d)]
                        if valid_distances:
                            # 初始化此N值的列表，如果不存在
                            if n not in metrics_results[metric_name][method]:
                                metrics_results[metric_name][method][n] = []
                            # 添加平均距离到结果字典
                            metrics_results[metric_name][method][n].append(np.mean(valid_distances))
                else:
                    # 对于其他方法，直接读取结果
                    analyze_file = f"{ANALYZE1_DIR}{sample_ratio}/{data_id:04}_{method}_analyze1.csv"
                    if os.path.exists(analyze_file):
                        analyze_data = pd.read_csv(analyze_file, index_col=0)
                        distance = analyze_data.loc[metric_name, 'Sample_0']
                        # 初始化此N值的列表，如果不存在
                        if n not in metrics_results[metric_name][method]:
                            metrics_results[metric_name][method][n] = []
                        # 添加距离到结果字典
                        metrics_results[metric_name][method][n].append(distance)

# 为每个指标计算不同几何体下的平均值
for metric_name, metric_results in metrics_results.items():
    # 计算每个采样方法、每个N值的平均值
    mean_results = {}
    for method in sample_methods:
        mean_results[method] = {}
        for n, values in metric_results[method].items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean_results[method][n] = np.mean(valid_values)
            else:
                mean_results[method][n] = np.nan
    
    # 创建DataFrame，索引为N值，列为采样方法
    df = pd.DataFrame(index=n_values)
    for method in sample_methods:
        df[method] = [mean_results[method].get(n, np.nan) for n in n_values]
    
    # 保存为CSV文件
    csv_filename = f'{OUTPUT_DIR}exp3_dim3_{metrics[metric_name]}_data.csv'
    df.to_csv(csv_filename)
    print(f"Saved data to {csv_filename}")

# 遍历每个指标
for metric_name, metric_file_name in metrics.items():
    # 创建一个 5 行 3 列的子图
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(14, 17))
    axes = axes.flatten()  # 展平 axes 以便按顺序使用

    # 遍历每个几何体
    for i, geometry in enumerate(geometries):
        # 获取对应的数据 ID
        geometry_data = data_info[data_info['Geometry'] == geometry]
        
        if geometry_data.empty:
            print(f"No data found for geometry: {geometry}")
            continue
            
        # 初始化存储结果的字典
        results = {method: [] for method in sample_methods}

        # 遍历所有N值和对应的x位置
        for n, x_pos in zip(n_values, x_positions):
            # 获取特定N值对应的数据行
            filtered_data = geometry_data[geometry_data['SampleNum'] == n]
            
            if filtered_data.empty:
                print(f"No data found for geometry {geometry} with N={n}")
                continue
                
            data_id = filtered_data['ID'].values[0]
            
            # 遍历每个采样方法
            for method in sample_methods:
                if method in ['srs', 'fps']:
                    # 对于 srs 和 fps，取三个随机数种子的均值
                    distances = []
                    for seed in random_seeds:
                        analyze_file = f"{ANALYZE1_DIR}{sample_ratio}/{data_id:04}_{method}_{seed}_analyze1.csv"
                        if os.path.exists(analyze_file):
                            analyze_data = pd.read_csv(analyze_file, index_col=0)
                            distance = analyze_data.loc[metric_name, 'Sample_0']
                            distances.append(distance)
                    
                    # 只有当至少有一个有效距离时才计算均值
                    if distances and not all(np.isnan(distances)):
                        # 过滤掉NaN值再计算均值
                        valid_distances = [d for d in distances if not np.isnan(d)]
                        if valid_distances:
                            results[method].append((x_pos, n, np.mean(valid_distances)))
                else:
                    # 对于其他方法，直接读取结果
                    analyze_file = f"{ANALYZE1_DIR}{sample_ratio}/{data_id:04}_{method}_analyze1.csv"
                    if os.path.exists(analyze_file):
                        analyze_data = pd.read_csv(analyze_file, index_col=0)
                        distance = analyze_data.loc[metric_name, 'Sample_0']
                        results[method].append((x_pos, n, distance))

        # 绘制折线图
        for j, method in enumerate(sample_methods):
            data_points = results[method]
            
            # 如果没有数据，则跳过
            if not data_points:
                continue
                
            # 按照x位置排序
            data_points.sort(key=lambda x: x[0])
            
            # 分离x位置、N值和指标值
            x_values = [item[0] for item in data_points]
            n_labels = [item[1] for item in data_points]
            y_values = [item[2] for item in data_points]
            
            # 检查是否有NaN值，并将其替换为该方法的平均值
            mask = np.isnan(y_values)
            if np.any(mask) and not np.all(mask):
                y_values = np.array(y_values)
                mean_val = np.nanmean(y_values)
                y_values[mask] = mean_val
                
            axes[i].plot(x_values, y_values, label=method, color=colors[j], marker=markers[j], linewidth=2)

        # 设置子图标题和标签
        axes[i].set_title(geometry, fontsize=14)
        axes[i].set_xlabel('N', fontsize=12)
        axes[i].set_ylabel('value', fontsize=12)
        
        # 设置x轴刻度和标签为实际的N值
        axes[i].set_xticks(x_positions)
        axes[i].set_xticklabels([str(n) for n in n_values], fontsize=10)
        
        axes[i].legend(loc='best', fontsize=10)
        axes[i].set_yscale('function', functions=(np.sqrt, np.square))
        
        # 不再使用对数尺度表示x轴，因为我们现在是等距排布
        # axes[i].set_xscale('log')

    # 大图标题，增大标题和子图间的距离
    fig.suptitle(metric_file_name, fontsize=20, y=1.01)

    # 调整子图间距
    plt.tight_layout(pad=1.0)  # 将四边距设置为 1.0
    plt.subplots_adjust(hspace=0.4, wspace=0.2)  # 调整行间距和列间距

    # 保存图像
    plt.savefig(f'{OUTPUT_DIR}exp3_dim3_{metric_file_name}.png', dpi=300, bbox_inches='tight')
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    print(f"Saved figure: {OUTPUT_DIR}exp3_dim3_{metric_file_name}.png")

print("Done!")