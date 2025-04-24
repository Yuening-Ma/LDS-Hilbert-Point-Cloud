import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置分析结果目录
ANALYZE1_DIR = 'dim_3/8_modelnet_2/analyze1_Mflexible/'  # 分析结果目录
OUTPUT_DIR = 'code_for_essay/output/exp4/'

# 定义模型名称列表
model_names = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair",
    "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar",
    # "keyboard", 
    "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", 
    "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", 
    "tv_stand", "vase", "wardrobe", "xbox"
]

# 定义采样方法
sample_methods = ['srs', 'fps', 'voxel', 'bds-pca', 'bds-hilbert']

# 定义颜色和标记
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'c']
markers = ['o', 's', '^', 'v', 'D', 'P', 'X']

# 设置采样率
sample_ratio = 0.20

# 定义要绘制的平滑指标
metrics = {
    'Hausdorff 95% Distance (Sample to GT)': 'HD95_sample_to_gt',
    'Hausdorff 95% Distance (GT to Sample)': 'HD95_gt_to_sample',
    'Chamfer Distance': 'chamfer_distance',
    'VFH Distance': 'vfh_distance',
    'Avg Distance to Shape': 'pc_to_mesh_distance'
}

# 遍历每个指标
for metric in metrics.keys():
    # 创建一个新的figure
    plt.figure(figsize=(14, 5))
    
    # 初始化存储结果的字典
    results = {method: [] for method in sample_methods}
    
    # 遍历每个模型
    for model_name in model_names:
        model_name = model_name + '_0001'
        
        # 遍历每个采样方法
        for method in sample_methods:
            if method in ['srs', 'fps']:
                # 对于 srs 和 fps，取三个随机数种子的均值
                distances = []
                for seed in [7, 42, 1309]:
                    analyze_file = f"{ANALYZE1_DIR}{int(sample_ratio * 100)}/{model_name}_{method}_{seed}_analyze1.csv"
                    analyze_data = pd.read_csv(analyze_file, index_col=0)
                    distance = analyze_data.loc[metric, 'Sample_0']
                    # 处理keyboard的异常值
                    if model_name == 'keyboard_0001':
                        distance = 0
                    distances.append(distance)

                if distances:
                    results[method].append(np.mean(distances))
            else:
                # 对于其他方法，直接读取结果
                analyze_file = f"{ANALYZE1_DIR}{int(sample_ratio * 100)}/{model_name}_{method}_analyze1.csv"
                analyze_data = pd.read_csv(analyze_file, index_col=0)
                distance = analyze_data.loc[metric, 'Sample_0']
                # 处理keyboard的异常值
                if model_name == 'keyboard_0001':
                    distance = 0
                results[method].append(distance)
    
    # 绘制折线图
    x = np.arange(len(model_names))
    
    for j, method in enumerate(sample_methods):
        if len(results[method]) == len(model_names):  # 确保数据完整
            plt.plot(x, results[method], label=method.upper(), color=colors[j], marker=markers[j], linestyle='-')
    
    # 设置图表属性
    plt.title(metric, fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('function', functions=(np.sqrt, np.square))
    
    # 设置x轴刻度
    plt.xticks(x, model_names, rotation=45, ha='right')
    
    # 添加图例到图表框内
    plt.legend(loc='upper right')
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    # 保存图像
    plt.savefig(f'{OUTPUT_DIR}/exp4_dim3_distance_{metrics[metric]}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/exp4_dim3_distance_{metrics[metric]}.svg', format='svg')
    plt.close()

print("Done!")