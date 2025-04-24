import numpy as np
import matplotlib.pyplot as plt

# 生成原始点云
n = 100
# pc = np.random.rand(n)
# np.savetxt('code_for_essay/data/dim1_pc_N100.csv', pc, delimiter=',')
pc = np.loadtxt('code_for_essay/data/dim1_pc_N100.csv', delimiter=',')

# 点云从小到大排序
pc_sorted = np.sort(pc)

# 生成BDS序列的ranks
trans_num = np.e
array = np.arange(1, n+1) * trans_num % 1
temp = np.argsort(array)
ranks = np.argsort(temp)

# 使用BDS生成的ranks给pc_sorted排序
pc_rank_sorted = pc_sorted[ranks]

# 随机取pc_rank_sorted的一段M个点（M=20）
M = 20
down_sampled_indices = np.random.choice(n, M, replace=False)
down_sampled_indices = np.sort(down_sampled_indices)  # 为了绘图方便，对索引排序
pc_down_sampled = pc_rank_sorted[down_sampled_indices]

# 创建4行1列的子图
fig, axs = plt.subplots(2, 2, figsize=(14, 6))
axs = axs.flatten()

# 第一个subplot：Original Point Cloud
axs[0].scatter(np.arange(n), pc, s=15, c='b', marker='o')
axs[0].set_title('Original Point Cloud', fontsize=12, pad=10)
axs[0].set_xlabel('Index', fontsize=10)
axs[0].set_ylabel('Value', fontsize=10)
axs[0].tick_params(axis='both', labelsize=10)

# 第二个subplot：Sorted Point Cloud
axs[1].scatter(np.arange(n), pc_sorted, s=15, c='g', marker='o')
axs[1].set_title('Sorted Point Cloud', fontsize=12, pad=10)
axs[1].set_xlabel('Index', fontsize=10)
axs[1].set_ylabel('Value', fontsize=10)
axs[1].tick_params(axis='both', labelsize=10)

# 第三个subplot：Point Cloud Sorted with Rank
axs[2].scatter(np.arange(n), pc_rank_sorted, s=15, c='r', marker='o')
axs[2].set_title('Point Cloud Sorted with Rank', fontsize=12, pad=10)
axs[2].set_xlabel('Index', fontsize=10)
axs[2].set_ylabel('Value', fontsize=10)
axs[2].tick_params(axis='both', labelsize=10)

# 第四个subplot：Down-Sampled Point Cloud
axs[3].scatter(np.arange(M), pc_down_sampled, s=15, c='c', marker='o')
axs[3].set_title('Down-Sampled Point Cloud', fontsize=12, pad=10)
axs[3].set_xlabel('Index', fontsize=10)
axs[3].set_ylabel('Value', fontsize=10)
axs[3].tick_params(axis='both', labelsize=10)

# 调整子图间距
plt.tight_layout(pad=1.0)  # 设置四边距为 1.0
plt.subplots_adjust(hspace=0.4, wspace=0.1)  # 调整子图之间的行间距

# 保存和显示图像
plt.savefig(f'code_for_essay/output/method_4.png', dpi=300)
plt.savefig(f'code_for_essay/output/method_4.svg', format='svg')

plt.show()