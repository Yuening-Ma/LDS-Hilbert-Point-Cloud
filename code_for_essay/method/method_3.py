import numpy as np
import matplotlib.pyplot as plt

# 生成BDS序列
n = 100
trans_num = np.e
array = np.arange(1, n+1) * trans_num % 1
temp = np.argsort(array)
ranks = np.argsort(temp)

# 创建1行2列的子图
fig, axs = plt.subplots(1, 2, figsize=(14, 3))

# 第一个subplot：array的散点图
axs[0].scatter(np.arange(1, n+1), array, s=25, c='b', marker='o')
axs[0].set_title('BDS', fontsize=12)
# axs[0].set_xlabel('Index', fontsize=10)
# axs[0].set_ylabel('Value', fontsize=10)
axs[0].tick_params(axis='both', labelsize=10)

# 第二个subplot：ranks的散点图
axs[1].scatter(np.arange(1, n+1), ranks, s=25, c='g', marker='o')
axs[1].set_title('BDS ranks', fontsize=12)
# axs[1].set_xlabel('Index', fontsize=10)
# axs[1].set_ylabel('Rank', fontsize=10)
axs[1].tick_params(axis='both', labelsize=10)

# 调整子图间距
plt.tight_layout(pad=1.0)  # 设置四边距为 1.0
plt.subplots_adjust(hspace=0.2, wspace=0.1)  # 调整子图之间的行间距和列间距

# 保存和显示图像
plt.savefig(f'code_for_essay/output/method_3.png', dpi=300)
plt.savefig(f'code_for_essay/output/method_3.svg', format='svg')

plt.show()