import os
import pandas as pd
import numpy as np
import sys
import time

# 添加自定义采样方法模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')

from sample_methods import *

M = 'int'
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'
SAMPLE_OUTPUT_DIR = f'dim_3/5_closed_geometry_2/sample_M{M}/'
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

RADIUS_NORMAL = 7.5

# 读取数据信息
data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

# 创建时间记录DataFrame
time_df = pd.DataFrame()

for index, row in data_info.iterrows():
    # 依次读取每一行的各个字段的值
    data_id = row['ID']
    geometry = row['Geometry'] 
    parameters = row['Parameters']
    data_points = row['DataPoints']
    
    # 打印读取的值
    print(f"ID: {data_id}, Geometry: {geometry}, Parameters: {parameters}, DataPoints: {data_points}")
    
    # 读取点云数据
    pc = np.loadtxt(f"{DATA_DIR}{data_id:04}_noised.csv", delimiter=',')
    N = pc.shape[0]

    assert N == data_points

    # for sample_ratio in [0.05, 0.10, 0.25]:
    for sample_ratio in [0.10]:
        num_sample = int(N * sample_ratio)
        num_cluster = int(1 / sample_ratio)
        print('\t', num_sample, num_cluster)
        os.makedirs(f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/', exist_ok=True)

        # SRS（简单随机抽样）
        for random_seed in [7, 42, 1309]:
            start_time = time.time()
            pc_sampled = random_sampling_cluster(
                pc, 
                num_cluster=num_cluster, 
                random_seed=random_seed
            )
            end_time = time.time()
            sampling_time = end_time - start_time
            
            # 记录时间
            method_name = f'srs_{random_seed}'
            time_df.loc[f"{data_id}_{sample_ratio}", method_name] = sampling_time
            
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
            start_time = time.time()
            pc_sampled = bds_sampling_cluster(
                pc, 
                num_cluster=num_cluster,
                trans_num=np.e, 
                sort=sort_method,
                dim_reduction='PCA'
            )
            end_time = time.time()
            sampling_time = end_time - start_time
            
            # 记录时间
            method_name = f'bds_{sort_method.lower()}'
            time_df.loc[f"{data_id}_{sample_ratio}", method_name] = sampling_time
            
            M, s = divmod(N, num_cluster)
            column_sizes = ([M + 1] * 3 * s + [M] * 3 * (num_cluster - s))  # 二维数据，每组两列
            header = np.array([column_sizes])
            np.savetxt(
                f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_bds-{sort_method.lower()}.csv', 
                np.vstack([header, pc_sampled]), 
                fmt='%.6f', 
                delimiter=','
            )

            # pc_sampled = bds_sampling_xu_v3(
            #     pc, 
            #     num_sample=num_sample, 
            #     radius=RADIUS_NORMAL, 
            #     k=30, 
            #     trans_num=np.e, 
            #     sort=sort_method, 
            #     dim_reduction='PCA'
            # )
            # header = np.array([[pc_sampled.shape[0], pc_sampled.shape[0], pc_sampled.shape[0]]])
            # np.savetxt(
            #     f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_bds-{sort_method.lower()}_xuv3.csv', 
            #     np.vstack([header, pc_sampled]), 
            #     fmt='%.6f', 
            #     delimiter=','
            # )

            # start_time = time.time()
            # pc_sampled = bds_sampling_xu_v4(
            #     pc, 
            #     num_sample=num_sample, 
            #     radius=RADIUS_NORMAL, 
            #     trans_num=np.e, 
            #     sort=sort_method, 
            #     dim_reduction='PCA', 
            #     edge_ratio=0.4, 
            #     sparse_ratio=2
            # )
            # end_time = time.time()
            # sampling_time = end_time - start_time
            
            # # 记录时间
            # method_name = f'bds_{sort_method.lower()}_xuv4'
            # time_df.loc[f"{data_id}_{sample_ratio}", method_name] = sampling_time
            
            # header = np.array([[pc_sampled.shape[0], pc_sampled.shape[0], pc_sampled.shape[0]]])
            # np.savetxt(
            #     f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_bds-{sort_method.lower()}_xuv4.csv', 
            #     np.vstack([header, pc_sampled]), 
            #     fmt='%.6f', 
            #     delimiter=','
            # )

        # Voxel（体素网格抽样）
        start_time = time.time()
        pc_sampled = voxel_sampling_flexible(
            pc, 
            num_sample=num_sample
        )
        end_time = time.time()
        sampling_time = end_time - start_time
        
        # 记录时间
        method_name = 'voxel'
        time_df.loc[f"{data_id}_{sample_ratio}", method_name] = sampling_time
        
        header = np.array([[pc_sampled.shape[0], pc_sampled.shape[0], pc_sampled.shape[0]]])
        np.savetxt(
            f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_voxel.csv', 
            np.vstack([header, pc_sampled]), 
            fmt='%.6f', 
            delimiter=','
        )

        # FPS（最远点采样）
        for random_seed in [7, 42, 1309]:
            start_time = time.time()
            pc_sampled = fps(
                pc, 
                num_sample=num_sample, 
                random_seed=random_seed
            )
            end_time = time.time()
            sampling_time = end_time - start_time
            
            # 记录时间
            method_name = f'fps_{random_seed}'
            time_df.loc[f"{data_id}_{sample_ratio}", method_name] = sampling_time
            
            header = np.array([[pc_sampled.shape[0], pc_sampled.shape[0], pc_sampled.shape[0]]])
            np.savetxt(
                f'{SAMPLE_OUTPUT_DIR}{int(sample_ratio * 100)}/{data_id:04}_fps_{random_seed}.csv', 
                np.vstack([header, pc_sampled]), 
                fmt='%.6f', 
                delimiter=','
            )

# 保存时间记录
time_df.to_csv(os.path.join(SAMPLE_OUTPUT_DIR, 'sampling_time.csv'))
print(f"\n采样时间记录已保存到: {os.path.join(SAMPLE_OUTPUT_DIR, 'sampling_time.csv')}")