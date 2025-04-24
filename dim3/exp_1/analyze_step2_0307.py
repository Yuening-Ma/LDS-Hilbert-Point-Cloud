import os
import pandas as pd
import numpy as np
import itertools
import ast

DATA_DIR = f'dim_3/4_surface_noised/data_Mint/'
ANALYZE1_DIR = f'dim_3/4_surface_noised/analyze1_Mint/'
ANALYZE2_DIR = f'dim_3/4_surface_noised/analyze2_Mint/'
os.makedirs(ANALYZE2_DIR, exist_ok=True)

random_seeds = [7, 42, 1309]
methods_cluster = [f"srs_{seed}" for seed in random_seeds] + ['bds-pca', 'bds-hilbert']

methods_need_seed = ['srs', 'fps']
methods_random_combinations = [f"{method}_{seed}" for method, seed in itertools.product(methods_need_seed, random_seeds)]
methods_all = methods_random_combinations + ['bds-pca', 'bds-hilbert', 'voxel']

data_info = pd.read_csv(DATA_DIR + 'data_info.csv')

for index, row in data_info.iterrows():
    data_id = row['ID']
    surface = row['Surface']
    parameters = row['Parameters']
    data_points = row['DataPoints']

    print(f"ID: {data_id}, Surface: {surface}, Parameters: {parameters}, DataPoints: {data_points}")

    for sample_ratio in [0.10, 0.20]:
        print(f"\tSample Ratio: {sample_ratio}")
        os.makedirs(f'{ANALYZE2_DIR}{int(sample_ratio*100)}/', exist_ok=True)

        '''
        对于分群的降采样方法，计算参数估计结果的群间方差
        '''
        # 初始化一个大的 DataFrame，用于存储所有方法的结果
        all_variances_df = pd.DataFrame(index=["Param 1 Variance", "Param 2 Variance", 
                                               "Param 3 Variance", "Param 4 Variance", 
                                               "Param 5 Variance", "Param 6 Variance"])

        for method in methods_cluster:
            analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"
            if not os.path.exists(analyze1_filename):
                print(f"\t\tFile not found: {analyze1_filename}")
                continue

            analyze1_df = pd.read_csv(analyze1_filename, index_col=0)

            # 提取相关行的数据
            estimated_params = analyze1_df.loc["Estimated Param 1"].values
            estimated_params_2 = analyze1_df.loc["Estimated Param 2"].values
            estimated_params_3 = analyze1_df.loc["Estimated Param 3"].values
            estimated_params_4 = analyze1_df.loc["Estimated Param 4"].values
            estimated_params_5 = analyze1_df.loc["Estimated Param 5"].values
            estimated_params_6 = analyze1_df.loc["Estimated Param 6"].values

            # 计算方差
            param_1_variance = np.var(estimated_params.astype(float))
            param_2_variance = np.var(estimated_params_2.astype(float))
            param_3_variance = np.var(estimated_params_3.astype(float))
            param_4_variance = np.var(estimated_params_4.astype(float))
            param_5_variance = np.var(estimated_params_5.astype(float))
            param_6_variance = np.var(estimated_params_6.astype(float))

            # 将结果添加到大的 DataFrame 中
            all_variances_df[method] = [param_1_variance, param_2_variance, 
                                        param_3_variance, param_4_variance, 
                                        param_5_variance, param_6_variance]

        # 保存汇总的 DataFrame 到文件
        output_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id:04}_cluster_variances.csv"
        all_variances_df.to_csv(output_filename, float_format="%.20f")
        print(f"\t\tSaved: {output_filename}")

        '''
        对于所有降采样方法，对各个衡量“差异”的项目，求均值，并统计 R2 > 0.99 的比例
        '''
        # 初始化一个大的 DataFrame，用于存储所有方法的偏差均值
        bias_df = pd.DataFrame(index=[
                                      "Param 1 Diff (Ground Truth) Mean", 
                                      "Param 2 Diff (Ground Truth) Mean", 
                                      "Param 3 Diff (Ground Truth) Mean", 
                                      "Param 4 Diff (Ground Truth) Mean", 
                                      "Param 5 Diff (Ground Truth) Mean", 
                                      "Param 6 Diff (Ground Truth) Mean", 
                                      "Param 1 Diff (Original) Mean", 
                                      "Param 2 Diff (Original) Mean", 
                                      "Param 3 Diff (Original) Mean", 
                                      "Param 4 Diff (Original) Mean", 
                                      "Param 5 Diff (Original) Mean", 
                                      "Param 6 Diff (Original) Mean", 
                                      ])

        for method in methods_all:
            analyze1_filename = f"{ANALYZE1_DIR}{int(sample_ratio*100)}/{data_id:04}_{method}_analyze1.csv"
            if not os.path.exists(analyze1_filename):
                print(f"\t\tFile not found: {analyze1_filename}")
                continue

            analyze1_df = pd.read_csv(analyze1_filename, index_col=0)

            # 提取相关行的数据
            kl_divergence_gt = analyze1_df.loc["KL Divergence (Ground Truth)"].values.astype(float)
            kl_divergence_original = analyze1_df.loc["KL Divergence (Original)"].values.astype(float)
            js_divergence_gt = analyze1_df.loc["JS Divergence (Ground Truth)"].values.astype(float)
            js_divergence_original = analyze1_df.loc["JS Divergence (Original)"].values.astype(float)
            param_diffs_gt = analyze1_df.loc["Param 1 Diff (Ground Truth)"].values.astype(float)
            param_diffs_original = analyze1_df.loc["Param 1 Diff (Original)"].values.astype(float)
            r2_values = analyze1_df.loc["R2"].values.astype(float)

            # 计算均值
            kl_divergence_gt_mean = np.mean(kl_divergence_gt)
            kl_divergence_original_mean = np.mean(kl_divergence_original)
            js_divergence_gt_mean = np.mean(js_divergence_gt)
            js_divergence_original_mean = np.mean(js_divergence_original)
            param_diffs_gt_mean = np.mean(param_diffs_gt)
            param_diffs_original_mean = np.mean(param_diffs_original)

            # 统计 R2 > 0.99 的比例
            r2_good_ratio = np.mean(r2_values > 0.95)

            # 将结果添加到 bias_df 中
            bias_df[method] = [
                               param_diffs_gt_mean, 
                               np.mean(analyze1_df.loc["Param 2 Diff (Ground Truth)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 3 Diff (Ground Truth)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 4 Diff (Ground Truth)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 5 Diff (Ground Truth)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 6 Diff (Ground Truth)"].values.astype(float)), 
                               param_diffs_original_mean, 
                               np.mean(analyze1_df.loc["Param 2 Diff (Original)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 3 Diff (Original)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 4 Diff (Original)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 5 Diff (Original)"].values.astype(float)), 
                               np.mean(analyze1_df.loc["Param 6 Diff (Original)"].values.astype(float)), 
                            #    r2_good_ratio
                               ]

        # 保存汇总的 bias_df 到文件
        bias_output_filename = f"{ANALYZE2_DIR}{int(sample_ratio*100)}/{data_id:04}_all_biases.csv"
        bias_df.to_csv(bias_output_filename, float_format="%.10f")
        print(f"\t\tSaved: {bias_output_filename}")