import os
import pandas as pd
import numpy as np
import sys

# 添加自定义采样方法模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')

from sample_methods import random_sampling_cluster, \
                            bds_sampling_cluster, \
                            fps, \
                            voxel_sampling_flexible

M = 'int'
TRANS_NUM = np.e
DATA_DIR = f'dim_3/4_surface_noised/data_M{M}/'
SAMPLE_OUTPUT_DIR = f'dim_3/4_surface_noised/sample_M{M}/'
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

for index, row in data_info.iterrows():
    # 依次读取每一行的各个字段的值
    data_id = row['ID']
    surface = row['Surface']
    parameters = row['Parameters']
    data_points = row['DataPoints']
    
    # 打印读取的值
    print(f"ID: {data_id}, Surface: {surface}, Parameters: {parameters}, DataPoints: {data_points}")
    
    # 读取点云数据
    pc = np.loadtxt(f"{DATA_DIR}{data_id:04}_noised.csv", delimiter=',')
    N = pc.shape[0]

    assert N == data_points

    for sample_ratio in [0.10, 0.20]:
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
            column_sizes = ([M + 1] * 3 * s + [M] * 3 * (num_cluster - s))  # 二维数据，每组两列
            # 生成表头
            header = np.array([column_sizes])
            # 保存结果
            np.savetxt(
                f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_srs_{random_seed}.csv', 
                np.vstack([header, pc_sampled]), 
                fmt='%.6f', 
                delimiter=','
            )

        # BDS
        for sort_method in ['PCA', 'Hilbert']:
            pc_sampled = bds_sampling_cluster(
                pc, 
                num_cluster=num_cluster,
                trans_num=TRANS_NUM,
                sort=sort_method
            )
            M, s = divmod(N, num_cluster)
            column_sizes = ([M + 1] * 3 * s + [M] * 3 * (num_cluster - s))  # 二维数据，每组两列
            header = np.array([column_sizes])
            np.savetxt(
                f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_bds-{sort_method.lower()}.csv', 
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
            f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_voxel.csv', 
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
                f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_fps_{random_seed}.csv', 
                np.vstack([header, pc_sampled]), 
                fmt='%.6f', 
                delimiter=','
            )