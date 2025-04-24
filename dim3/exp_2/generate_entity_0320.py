import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import open3d as o3d

# 添加自定义采样方法模块路径
sys.path.append('/home/yuening/2_scholar_projects/20250217_pc_sample/dim_3')
from geometry_3d import *

# 定义路径和参数
M = 'int'
DATA_DIR = f'dim_3/5_closed_geometry_2/data_M{M}/'
os.makedirs(DATA_DIR, exist_ok=True)

# 定义参数
n_values = [9180, 37240, 95760]
SIGMA = 1.0
SCALE = 100

# 定义几何体参数
closed_geometries = {
    "Sphere": (SCALE * 0.5,),  # 半径
    "Cylinder": (SCALE * 0.3, SCALE),  # 半径和高度
    "EquilateralTriangularPrism": (SCALE * 0.6, SCALE),  # 边长和高度
    # "IsoscelesRightTriangularPrism": (SCALE * 0.6, SCALE),  # 边长和高度
    "Cube": (SCALE,),  # 边长
    "QuadrangularPrism": (SCALE * 0.4, SCALE * 0.8, SCALE),  # 长、宽和高度
    "PentagonalPrism": (SCALE * 0.4, SCALE),  # 边长和高度
    "HexagonalPrism": (SCALE * 0.4, SCALE),  # 边长和高度
    "LBeam": (SCALE * 0.6, SCALE, SCALE * 0.1),  # 宽度、高度和厚度
    "TBeam": (SCALE * 0.6, SCALE, SCALE * 0.1),  # 宽度、高度和厚度
    "UBeam": (SCALE * 0.6, SCALE, SCALE * 0.1),  # 宽度、高度和厚度
    "HBeam": (SCALE * 0.6, SCALE, SCALE * 0.1),  # 宽度、高度和厚度
    "Cone": (SCALE * 0.5, SCALE),  # 半径和高度
    "Torus": (SCALE * 0.35, SCALE * 0.15),  # 环半径和管半径
    "Tetrahedron": (SCALE * 0.75,),  # 半径
    "Octahedron": (SCALE * 0.5,),  # 半径
    # "Mobius": (1, 1, 1, 1, SCALE)  # 半径、扁平度和宽度
}

# 生成并旋转所有几何体的mesh
print("Generating and rotating meshes...")
meshes = {}
for geometry_name, params in closed_geometries.items():
    print(f"Processing {geometry_name}...")
    # 生成几何体
    if geometry_name == "Sphere":
        mesh = create_sphere(*params)
    elif geometry_name == "Cylinder":
        mesh = create_cylinder(*params)
    elif geometry_name == "EquilateralTriangularPrism":
        mesh = create_equilateral_triangular_prism(*params)
    elif geometry_name == "IsoscelesRightTriangularPrism":
        mesh = create_isosceles_right_triangular_prism(*params)
    elif geometry_name == "Cube":
        mesh = create_cube(*params)
    elif geometry_name == "QuadrangularPrism":
        mesh = create_quadrangular_prism(*params)
    elif geometry_name == "PentagonalPrism":
        mesh = create_pentagonal_prism(*params)
    elif geometry_name == "HexagonalPrism":
        mesh = create_hexagonal_prism(*params)
    elif geometry_name == "LBeam":
        mesh = create_l_beam(*params)
    elif geometry_name == "TBeam":
        mesh = create_t_beam(*params)
    elif geometry_name == "UBeam":
        mesh = create_u_beam(*params)
    elif geometry_name == "HBeam":
        mesh = create_h_beam(*params)
    elif geometry_name == "Cone":
        mesh = create_cone(*params)
    elif geometry_name == "Torus":
        mesh = create_torus(*params)
    elif geometry_name == "Tetrahedron":
        mesh = create_tetrahedron(*params)
    elif geometry_name == "Octahedron":
        mesh = create_octahedron(*params)
    elif geometry_name == "Mobius":
        mesh = create_mobius(*params)

    # 随机旋转
    # 生成随机旋转角度（弧度）
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)
    
    # 创建旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    # 组合旋转矩阵
    R = Rx @ Ry @ Rz
    
    # 应用旋转
    mesh.rotate(R, center=mesh.get_center())

    # # 可视化旋转后的网格
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(mesh)
    # vis.run()
    # vis.destroy_window()

    # 存储mesh
    meshes[geometry_name] = mesh

# 数据信息
data_info = []
data_id = 0

# 遍历不同的数据量
for n in n_values:
    print(f"Processing for sample size: {n}")
    # 遍历几何体
    for geometry_name, mesh in meshes.items():
        # 保存网格
        mesh_path = f"{DATA_DIR}{data_id:04}.ply"
        o3d.io.write_triangle_mesh(mesh_path, mesh)

        # 生成点云
        points = generate_pc_from_mesh(mesh, sample_method="poisson", pc_num=n, sigma=None)
        # 生成噪声并限制在±3范围内
        noise = np.random.normal(0, SIGMA, size=points.shape)
        # 将超出±3范围的噪声值设为0
        # noise[np.abs(noise) > 3] = 0
        points_noised = points + noise

        # 保存点云
        np.savetxt(f"{DATA_DIR}{data_id:04}.csv", points, delimiter=",", fmt="%.6f")
        np.savetxt(f"{DATA_DIR}{data_id:04}_noised.csv", points_noised, delimiter=",", fmt="%.6f")

        # 记录数据信息
        params = closed_geometries[geometry_name]
        params_str = "_".join(map(str, params))
        data_info.append([data_id, geometry_name, params_str, n, len(points)])
        data_id += 1

# 保存数据信息
info_df = pd.DataFrame(data_info, columns=["ID", "Geometry", "Parameters", "SampleNum", "DataPoints"])
info_df.to_csv(f"{DATA_DIR}data_info.csv", index=False)

print("数据生成和保存完成！")