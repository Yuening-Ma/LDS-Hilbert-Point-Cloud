import numpy as np
import matplotlib.pyplot as plt
import sys

# 添加自定义采样方法模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/code_for_essay')
from sample_methods_2d import *

# 生成点云
def generate_point_cloud(n, distribution='normal'):
    if distribution == 'normal':
        # 二维正态分布点云
        point_cloud = np.random.normal(0, 1, (n, 2))
    elif distribution == 'circle':
        # 圆形点云
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        noise = np.random.normal(0, 0.05, (n, 2))  # 添加正态分布噪声
        point_cloud = np.vstack((x, y)).T + noise
    elif distribution == 'circle_dense':
        # 圆形点云，部分区域稠密，部分区域稀疏
        # x 均匀分布
        x = np.linspace(-1, 1, n)
        # 计算对应的 y 值，保留正负两组值
        y_positive = np.sqrt(1 - x**2)
        y_negative = -np.sqrt(1 - x**2)
        
        # 将正负两组 y 值合并
        y = np.concatenate((y_positive, y_negative))
        x = np.concatenate((x, x))
        
        # # 打乱顺序以避免明显的分层
        # shuffled_indices = np.random.permutation(len(x))
        # x = x[shuffled_indices]
        # y = y[shuffled_indices]
        
        # 添加正态分布噪声
        noise = np.random.normal(0, 0.05, (2 * n, 2))
        point_cloud = np.vstack((x, y)).T + noise
    else:
        raise ValueError("Invalid distribution type. Choose 'normal', 'circle', or 'circle_dense'.")
    return point_cloud

# 降采样比例
sampling_ratio = 0.1
num_cluster = int(1/sampling_ratio)
num_sample = 12  # 固定采样数量

# 创建3行6列的子图
fig, axs = plt.subplots(3, 6, figsize=(14, 7))
axs = axs.flatten()

point_clouds = []

# 第一行：二维正态分布点云
point_cloud = generate_point_cloud(120, distribution='normal')
point_clouds.append(point_cloud)
axs[0].scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='#007BFF', marker='o')
axs[0].set_title('Original Point Cloud', fontsize=12)
# axs[0].set_xlabel('X', fontsize=10)
axs[0].set_ylabel('Y', fontsize=10)
axs[0].tick_params(axis='both', labelsize=10)
axs[0].set_xlim(-3, 3)
axs[0].set_ylim(-3, 3)

# 第二行：圆形点云
point_cloud = generate_point_cloud(120, distribution='circle')
point_clouds.append(point_cloud)
axs[6].scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='#007BFF', marker='o')
# axs[6].set_title('Original Point Cloud', fontsize=12)
# axs[6].set_xlabel('X', fontsize=10)
axs[6].set_ylabel('Y', fontsize=10)
axs[6].tick_params(axis='both', labelsize=10)
axs[6].set_xlim(-1.2, 1.2)
axs[6].set_ylim(-1.2, 1.2)

# 第三行：圆形点云（部分区域稠密）
point_cloud = generate_point_cloud(120, distribution='circle_dense')
point_clouds.append(point_cloud)
axs[12].scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='#007BFF', marker='o')
# axs[12].set_title('Original Point Cloud', fontsize=12)
axs[12].set_xlabel('X', fontsize=10)
axs[12].set_ylabel('Y', fontsize=10)
axs[12].tick_params(axis='both', labelsize=10)
axs[12].set_xlim(-1.2, 1.2)
axs[12].set_ylim(-1.2, 1.2)

# 降采样方法
sampling_methods = [
    ('SRS', random_sampling_cluster, {'num_cluster': num_cluster, 'random_seed': 7}),
    ('FPS', fps, {'num_sample': num_sample, 'random_seed': 7}),
    ('Voxel', voxel_sampling_flexible, {'num_sample': num_sample}),
    ('BDS_PCA', bds_sampling_cluster, {'num_cluster': num_cluster, 'sort': 'PCA'}),
    ('BDS_Hilbert', bds_sampling_cluster, {'num_cluster': num_cluster, 'sort': 'Hilbert'})
]

# 应用降采样方法
for i, (name, func, kwargs) in enumerate(sampling_methods):
    # 第一行
    sampled_points = func(point_clouds[0], **kwargs)
    axs[i+1].scatter(sampled_points[:, 0], sampled_points[:, 1], s=10, c='#FF6347', marker='o')
    axs[i+1].set_title(name, fontsize=12)
    # axs[i+1].set_xlabel('X', fontsize=10)
    # axs[i+1].set_ylabel('Y', fontsize=10)
    axs[i+1].tick_params(axis='both', labelsize=10)
    axs[i+1].set_xlim(-3, 3)
    axs[i+1].set_ylim(-3, 3)

    # 第二行
    sampled_points = func(point_clouds[1], **kwargs)
    axs[i+7].scatter(sampled_points[:, 0], sampled_points[:, 1], s=10, c='#FF6347', marker='o')
    # axs[i+7].set_title(name, fontsize=12)
    # axs[i+7].set_xlabel('X', fontsize=10)
    # axs[i+7].set_ylabel('Y', fontsize=10)
    axs[i+7].tick_params(axis='both', labelsize=10)
    axs[i+7].set_xlim(-1.2, 1.2)
    axs[i+7].set_ylim(-1.2, 1.2)

    # 第三行
    sampled_points = func(point_clouds[2], **kwargs)
    axs[i+13].scatter(sampled_points[:, 0], sampled_points[:, 1], s=10, c='#FF6347', marker='o')
    # axs[i+13].set_title(name, fontsize=12)
    axs[i+13].set_xlabel('X', fontsize=10)
    # axs[i+13].set_ylabel('Y', fontsize=10)
    axs[i+13].tick_params(axis='both', labelsize=10)
    axs[i+13].set_xlim(-1.2, 1.2)
    axs[i+13].set_ylim(-1.2, 1.2)

# 调整子图间距
plt.tight_layout(pad=1.0)  # 设置四边距为 1.0
plt.subplots_adjust(hspace=0.25, wspace=0.25)  # 调整子图之间的行间距和列间距

# 保存和显示图像
plt.savefig(f'code_for_essay/output/method_5.png', dpi=300)
plt.savefig(f'code_for_essay/output/method_5.svg', format='svg')

plt.show()