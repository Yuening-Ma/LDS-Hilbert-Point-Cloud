import os
import numpy as np
import sys

# 添加自定义采样方法模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')
from sample_methods import *

M = 'flexible'
DATA_DIR = f'datasets/ModelNet40_2/'
SAMPLE_OUTPUT_DIR = f'dim_3/8_modelnet_2/sample_M{M}/'
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

# 遍历 DATA_DIR 中的 csv 文件
for file_name in sorted(os.listdir(DATA_DIR + 'pc_noise/')):
    if file_name.endswith('.csv'):
        # 获取文件名（不带扩展名）作为 data_id
        data_id = os.path.splitext(file_name)[0]
        
        # 打印当前处理的文件
        print(f"Processing file: {file_name}")
        
        # 读取点云数据
        pc = np.loadtxt(os.path.join(DATA_DIR + 'pc_noise/', file_name), delimiter=',')
        N = pc.shape[0]
        
        # for sample_ratio in [0.05, 0.10, 0.25]:
        for sample_ratio in [0.20]:
            num_sample = int(N * sample_ratio)
            num_cluster = int(1 / sample_ratio)
            print('\t', num_sample, num_cluster)
            os.makedirs(f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/', exist_ok=True)

            # SRS（简单随机抽样）
            for random_seed in [7, 42, 1309]:
                pc_sampled = random_sampling_cluster(
                    pc, 
                    num_cluster=num_cluster, 
                    random_seed=random_seed
                )
                # 计算每组的样本数量
                M, s = divmod(N, num_cluster)
                column_sizes = ([M + 1] * 3 * s + [M] * 3 * (num_cluster - s))  # 三维数据，每组三列
                # 生成表头
                header = np.array([column_sizes])
                # 保存结果
                np.savetxt(
                    f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id}_srs_{random_seed}.csv', 
                    np.vstack([header, pc_sampled]), 
                    fmt='%.6f', 
                    delimiter=','
                )

            # BDS
            for sort_method in ['PCA', 'Hilbert']:
                pc_sampled = bds_sampling_cluster(
                    pc, 
                    num_cluster=num_cluster,
                    trans_num=np.e,  # 使用自然常数 e 作为转换系数
                    sort=sort_method,
                )
                M, s = divmod(N, num_cluster)
                column_sizes = ([M + 1] * 3 * s + [M] * 3 * (num_cluster - s))  # 三维数据，每组三列
                header = np.array([column_sizes])
                np.savetxt(
                    f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id}_bds-{sort_method.lower()}.csv', 
                    np.vstack([header, pc_sampled]), 
                    fmt='%.6f', 
                    delimiter=','
                )

            # Voxel（体素网格抽样）
            pc_sampled = voxel_sampling_flexible(
                pc, 
                num_sample=num_sample
            )
            header = np.array([[pc_sampled.shape[0], pc_sampled.shape[0], pc_sampled.shape[0]]])
            np.savetxt(
                f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id}_voxel.csv', 
                np.vstack([header, pc_sampled]), 
                fmt='%.6f', 
                delimiter=','
            )

            # FPS（最远点采样）
            for random_seed in [7, 42, 1309]:
                pc_sampled = fps(
                    pc, 
                    num_sample=num_sample, 
                    random_seed=random_seed
                )
                header = np.array([[pc_sampled.shape[0], pc_sampled.shape[0], pc_sampled.shape[0]]])
                np.savetxt(
                    f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id}_fps_{random_seed}.csv', 
                    np.vstack([header, pc_sampled]), 
                    fmt='%.6f', 
                    delimiter=','
                )