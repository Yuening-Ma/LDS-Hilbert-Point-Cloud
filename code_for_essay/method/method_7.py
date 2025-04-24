import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import pandas as pd
from scipy.spatial.distance import directed_hausdorff


# 添加自定义采样方法模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/code_for_essay')
from sample_methods_2d import *

# 二维版本的点云距离，即从 pc_2 为每个 pc_1 找到最近点并计算距离    
def point_cloud_distance(pc_1, pc_2):
    # 创建KD树
    tree = cKDTree(pc_2)
    
    # 为每个 pc_1 找到最近点并计算距离
    distances, _ = tree.query(pc_1, k=1, p=2)
    return distances

# 二维点云的 chamfer distance，即pc_1到pc_2距离的均值，加上pc_2到pc_1距离的均值
def chamfer_distance(pc_1, pc_2):
    distances_1 = point_cloud_distance(pc_1, pc_2)
    distances_2 = point_cloud_distance(pc_2, pc_1)
    return np.mean(distances_1) + np.mean(distances_2)

'''
基于直方图的散度
'''
def kl_divergence_hist(pc_1, pc_2, num_bins=10):
    # 计算二维直方图
    x_min, y_min = min(pc_1[:, 0].min(), pc_2[:, 0].min()), min(pc_1[:, 1].min(), pc_2[:, 1].min())
    x_max, y_max = max(pc_1[:, 0].max(), pc_2[:, 0].max()), max(pc_1[:, 1].max(), pc_2[:, 1].max())

    hist_1, _, _ = np.histogram2d(pc_1[:, 0], pc_1[:, 1], bins=num_bins, range=[[x_min, x_max], [y_min, y_max]])
    hist_2, _, _ = np.histogram2d(pc_2[:, 0], pc_2[:, 1], bins=num_bins, range=[[x_min, x_max], [y_min, y_max]])

    # 归一化直方图
    hist_1 = hist_1 / hist_1.sum()
    hist_2 = hist_2 / hist_2.sum()

    # 避免零值导致的数值问题
    epsilon = 1e-10
    hist_1 = np.maximum(hist_1, epsilon)
    hist_2 = np.maximum(hist_2, epsilon)

    # 计算KL散度
    kl_divergence = stats.entropy(hist_1.flatten(), hist_2.flatten())
    return kl_divergence


# 生成点云
def generate_point_cloud(n, distribution='circle_dense'):
    if distribution == 'circle_dense':
        # 圆形点云，部分区域稠密，部分区域稀疏
        # x 均匀分布
        x = np.linspace(-1, 1, int(n/2))
        # 计算对应的 y 值，保留正负两组值
        y_positive = np.sqrt(1 - x**2)
        y_negative = -np.sqrt(1 - x**2)
        
        # 将正负两组 y 值合并
        y = np.concatenate((y_positive, y_negative))
        x = np.concatenate((x, x))
        
        # 添加正态分布噪声
        noise = np.random.normal(0, 0.02, (n, 2))
        pc_gt = np.vstack((x, y)).T
        pc = np.vstack((x, y)).T + noise
    else:
        raise ValueError("Invalid distribution type. Choose 'circle_dense'.")
    return pc_gt, pc

# 降采样比例
n = 1230
sampling_ratio = 0.1
num_cluster = int(1/sampling_ratio)
num_sample = int(n * sampling_ratio)  # 固定采样数量

# 创建2行4列的子图
fig, axs = plt.subplots(2, 4, figsize=(14, 7))
axs = axs.flatten()

# # 生成圆形点云（部分区域稠密）
# pc_gt, pc = generate_point_cloud(n, distribution='circle_dense')
# # 保存原始点云
# np.savetxt(f'code_for_essay/output/method/pc_gt_{n}.csv', pc_gt, delimiter=',')
# np.savetxt(f'code_for_essay/output/method/pc_{n}.csv', pc, delimiter=',')

# 从csv文件读取点云
pc_gt = np.loadtxt(f'code_for_essay/output/method/pc_gt_{n}.csv', delimiter=',')
pc = np.loadtxt(f'code_for_essay/output/method/pc_{n}.csv', delimiter=',')

# 绘制原始点云（无噪声）
axs[0].scatter(pc_gt[:, 0], pc_gt[:, 1], s=5, c='#007BFF', marker='o')
axs[0].set_title('Ground Truth', fontsize=12)
axs[0].set_xlabel('X', fontsize=10)
axs[0].set_ylabel('Y', fontsize=10)
axs[0].tick_params(axis='both', labelsize=10)
axs[0].set_xlim(-1.2, 1.2)
axs[0].set_ylim(-1.2, 1.2)
# 保持比例
axs[0].set_aspect('equal', 'box')

# 绘制原始点云（带噪声）
axs[1].scatter(pc[:, 0], pc[:, 1], s=5, c='#007BFF', marker='o')
axs[1].set_title('Original Point Cloud', fontsize=12)
axs[1].set_xlabel('X', fontsize=10)
axs[1].set_ylabel('Y', fontsize=10)
axs[1].tick_params(axis='both', labelsize=10)
axs[1].set_xlim(-1.2, 1.2)
axs[1].set_ylim(-1.2, 1.2)
# 保持比例
axs[1].set_aspect('equal', 'box')

# 降采样方法
sampling_methods = [
    ('SRS', random_sampling_cluster, {'num_cluster': num_cluster, 'random_seed': 7}),
    ('FPS', fps, {'num_sample': num_sample, 'random_seed': 7}),
    ('Voxel', voxel_sampling_flexible, {'num_sample': num_sample}),
    ('LDS_PCA', bds_sampling_cluster, {'num_cluster': num_cluster, 'sort': 'PCA'}),
    ('LDS_Hilbert', bds_sampling_cluster, {'num_cluster': num_cluster, 'sort': 'Hilbert'})
]

# 应用降采样方法
for i, (name, func, kwargs) in enumerate(sampling_methods):
    sampled_pc = func(pc, **kwargs)
    axs[i+2].scatter(sampled_pc[:, 0], sampled_pc[:, 1], s=10, c='#FF6347', marker='o')
    axs[i+2].set_title(name, fontsize=12)
    axs[i+2].set_xlabel('X', fontsize=10)
    axs[i+2].set_ylabel('Y', fontsize=10)
    axs[i+2].tick_params(axis='both', labelsize=10)
    axs[i+2].set_xlim(-1.2, 1.2)
    axs[i+2].set_ylim(-1.2, 1.2)
    # 保持比例
    axs[i+2].set_aspect('equal', 'box')

# 隐藏最后一个子图
axs[-1].axis('off')

# 调整子图间距
plt.tight_layout(pad=1.0)  # 设置四边距为 1.0
plt.subplots_adjust(hspace=0.25, wspace=0.25)  # 调整子图之间的行间距和列间距

# 保存和显示图像
plt.savefig(f'code_for_essay/output/method/method_7.png', dpi=300)
plt.savefig(f'code_for_essay/output/method/method_7.svg', format='svg')

plt.show()

# 为每个降采样方法计算点云距离，chamfer distance和kl散度，写为pd.DataFrame，保存为csv

# 创建一个空的DataFrame，行名称为各个指标，列名称为方法名称
df = pd.DataFrame(index=['pc_distance_1', 'pc_distance_2', 'chamfer_distance', 'kl_divergence'])

for i, (name, func, kwargs) in enumerate(sampling_methods):
    sampled_pc = func(pc, **kwargs)[:, :2]
    dist_1 = np.mean(point_cloud_distance(sampled_pc, pc_gt))
    dist_2 = np.mean(point_cloud_distance(pc_gt, sampled_pc))
    cd = chamfer_distance(sampled_pc, pc_gt)
    kl = kl_divergence_hist(sampled_pc, pc_gt, num_bins=10)

    # 添加到DataFrame，列名称为方法名称
    df[name] = [dist_1, dist_2, cd, kl]

# 保存为csv
df.to_csv(f'code_for_essay/output/method/method_7_result.csv', index=True)
