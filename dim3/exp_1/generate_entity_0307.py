import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义高斯噪声的标准差
NOISE_STD = 0.05  # 噪声的标准差

# 定义生成数据的数量
n_values = [9180, 37240, 95760]  # 总点数
n_decompositions = [(90, 102), (196, 190), (304, 315)]  # 分解为两个自由度的点数

# 定义每种参数曲面的参数组合
surfaces = {
    "Plane": [(1, 0.5, 1.5)],  # 平面参数：(a, b, c) 对应 z=ax+by+c
    "Sphere": [(0, 0, 0, 1)],  # 球面参数：(x_center, y_center, z_center, radius)
    "Ellipsoid": [(0, 0, 0, 3, 2, 1)],  # 椭球面参数：(x_center, y_center, z_center, a, b, c)
    "Torus": [(0, 0, 0, 2, 0.5)],  # 环面参数：(x_center, y_center, z_center, R, r)
    "Cylinder": [(0, 0, 1)],  # 圆柱面参数：(x_center, y_center, radius)
    "Cone": [(0, 0, 0, 1)],  # 圆锥面参数：(x_center, y_center, z_center, radius)
    "Paraboloid": [(0, 0, 0, 0.5)]  # 抛物面参数：(x_center, y_center, z_center, a)
}

# 更新数据目录
DATA_DIR = f'dim_3/4_surface_noised/data_Mint/'
os.makedirs(DATA_DIR, exist_ok=True)

# 生成数据并保存到 CSV 文件
data_info = []  # 用于保存数据信息
data_id = 0  # 数据编号

for n, (n1, n2) in zip(n_values, n_decompositions):  # 遍历不同的数据量
    for surface_name, params_list in surfaces.items():
        for params in params_list:
            # 生成数据
            if surface_name == "Plane":
                # 平面参数方程
                a, b, c = params  # 参数 (a, b, c) 对应 z = ax + by + c
                x = np.linspace(-2, 2, n1)
                y = np.linspace(-2, 2, n2)
                x, y = np.meshgrid(x, y)
                z = a * x + b * y + c  # 计算 z 值
                x, y, z = x.flatten(), y.flatten(), z.flatten()
            elif surface_name == "Sphere":
                # 球面参数方程
                x_center, y_center, z_center, radius = params
                theta = np.linspace(0, 2 * np.pi, n1)
                phi = np.linspace(0, np.pi, n2)
                theta, phi = np.meshgrid(theta, phi)
                x = x_center + radius * np.sin(phi) * np.cos(theta)
                y = y_center + radius * np.sin(phi) * np.sin(theta)
                z = z_center + radius * np.cos(phi)
                x, y, z = x.flatten(), y.flatten(), z.flatten()
            elif surface_name == "Ellipsoid":
                # 椭球面参数方程
                x_center, y_center, z_center, a, b, c = params
                theta = np.linspace(0, 2 * np.pi, n1)
                phi = np.linspace(0, np.pi, n2)
                theta, phi = np.meshgrid(theta, phi)
                x = x_center + a * np.sin(phi) * np.cos(theta)
                y = y_center + b * np.sin(phi) * np.sin(theta)
                z = z_center + c * np.cos(phi)
                x, y, z = x.flatten(), y.flatten(), z.flatten()
            elif surface_name == "Torus":
                # 环面参数方程
                x_center, y_center, z_center, R, r = params
                theta = np.linspace(0, 2 * np.pi, n1)
                phi = np.linspace(0, 2 * np.pi, n2)
                theta, phi = np.meshgrid(theta, phi)
                x = x_center + (R + r * np.cos(phi)) * np.cos(theta)
                y = y_center + (R + r * np.cos(phi)) * np.sin(theta)
                z = z_center + r * np.sin(phi)
                x, y, z = x.flatten(), y.flatten(), z.flatten()
            elif surface_name == "Cylinder":
                # 圆柱面参数方程
                x_center, y_center, radius = params
                z_center = 0
                theta = np.linspace(0, 2 * np.pi, n1)
                z = np.linspace(-2, 2, n2)
                theta, z = np.meshgrid(theta, z)
                x = x_center + radius * np.cos(theta)
                y = y_center + radius * np.sin(theta)
                z = z_center + z
                x, y, z = x.flatten(), y.flatten(), z.flatten()
            elif surface_name == "Cone":
                # 圆锥面参数方程
                x_center, y_center, z_center, radius = params
                theta = np.linspace(0, 2 * np.pi, n1)
                z = np.linspace(0, 2, n2)
                theta, z = np.meshgrid(theta, z)
                x = x_center + radius * z * np.cos(theta)
                y = y_center + radius * z * np.sin(theta)
                z = z_center + z
                x, y, z = x.flatten(), y.flatten(), z.flatten()
            elif surface_name == "Paraboloid":
                # 抛物面参数方程
                x_center, y_center, z_center, a = params
                r = np.linspace(0, 2, n1)
                theta = np.linspace(0, 2 * np.pi, n2)
                r, theta = np.meshgrid(r, theta)
                x = x_center + r * np.cos(theta)
                y = y_center + r * np.sin(theta)
                z = z_center + a * r**2
                x, y, z = x.flatten(), y.flatten(), z.flatten()

            # 保存干净的点云
            np.savetxt(f"{DATA_DIR}{data_id:04}.csv", np.column_stack((x, y, z)), delimiter=",", fmt="%.6f")

            # 添加高斯噪声
            noise_x = np.random.normal(0, NOISE_STD, size=x.shape)
            noise_y = np.random.normal(0, NOISE_STD, size=y.shape)
            noise_z = np.random.normal(0, NOISE_STD, size=z.shape)
            
            x_noised = x + noise_x
            y_noised = y + noise_y
            z_noised = z + noise_z

            # # 可视化三维点云
            # fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(121, projection='3d')  # 创建一个三维坐标轴
            # ax.scatter(x, y, z, s=2)  # s 是点的大小
            # ax.set_title(f"{surface_name} (n={n})", fontsize=14)
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # ax.set_box_aspect([1, 1, 1])  # 保持比例

            # ax = fig.add_subplot(122, projection='3d')  # 创建一个三维坐标轴
            # ax.scatter(x_noised, y_noised, z_noised, s=2)  # s 是点的大小
            # ax.set_title(f"noised {surface_name} (n={n})", fontsize=14)
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # ax.set_box_aspect([1, 1, 1])  # 保持比例

            # # 显示图形
            # plt.show()

            # 将 x, y, z 合并为三维点云
            data = np.column_stack((x_noised, y_noised, z_noised))

            # 保存数据到 CSV 文件（使用 numpy.savetxt）
            np.savetxt(f"{DATA_DIR}{data_id:04}_noised.csv", data, delimiter=",", fmt="%.6f")

            # 保存数据信息
            params_str = "_".join(map(str, params))  # 参数拼接成字符串
            data_info.append([data_id, surface_name, params_str, n])  # 添加数据量 n

            data_id += 1  # 更新编号

# 保存数据信息到 CSV 文件
info_df = pd.DataFrame(data_info, columns=["ID", "Surface", "Parameters", "DataPoints"])
info_df.to_csv(f"{DATA_DIR}data_info.csv", index=False)

print("噪声点云数据生成和保存完成！")