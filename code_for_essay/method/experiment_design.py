# 绘制一个简单的桌子:上面是一个扁圆柱(桌面),下面是四个细长的长方体(桌腿)
import open3d as o3d
import numpy as np

# 创建桌面（扁圆柱）
table_top = o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=0.5)
# 将桌面平移到合适的高度
table_top.translate([0, 0, 6])

# 创建桌腿（四个细长的长方体）
leg1 = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg1.translate([-3, -3, 0])  # 左下角桌腿

leg2 = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg2.translate([3, -3, 0])   # 右下角桌腿

leg3 = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg3.translate([-3, 3, 0])   # 左上角桌腿

leg4 = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg4.translate([3, 3, 0])    # 右上角桌腿

# 合并所有几何体
table = table_top
table += leg1
table += leg2
table += leg3
table += leg4

# 计算法向量
table.compute_vertex_normals()

# 设置浅蓝色
light_blue = np.array([0.7, 0.8, 1.0])
table.paint_uniform_color(light_blue)

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加网格到可视化器
vis.add_geometry(table)

# 设置渲染选项
opt = vis.get_render_option()
opt.background_color = np.asarray([1, 1, 1])  # 白色背景
opt.point_size = 1.0
opt.line_width = 1.0  # 设置线宽
opt.mesh_show_wireframe = True  # 显示线框
opt.mesh_show_back_face = True  # 显示背面

# 运行可视化
vis.run()
vis.destroy_window()

# 创建拆解视图
# 创建桌面（扁圆柱）
table_top_exploded = o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=0.5)
# 将桌面向上移动
table_top_exploded.translate([0, 0, 7])

# 创建桌腿（四个细长的长方体）
leg1_exploded = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg1_exploded.translate([-4, -4, 0])  # 左下角桌腿

leg2_exploded = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg2_exploded.translate([4, -4, 0])   # 右下角桌腿

leg3_exploded = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg3_exploded.translate([-4, 4, 0])   # 左上角桌腿

leg4_exploded = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=6)
leg4_exploded.translate([4, 4, 0])    # 右上角桌腿

# 计算法向量
table_top_exploded.compute_vertex_normals()
leg1_exploded.compute_vertex_normals()
leg2_exploded.compute_vertex_normals()
leg3_exploded.compute_vertex_normals()
leg4_exploded.compute_vertex_normals()

# 设置不同颜色
table_top_exploded.paint_uniform_color(np.array([0.7, 0.8, 1.0]))  # 浅蓝色桌面
leg1_exploded.paint_uniform_color(np.array([0.8, 0.7, 1.0]))  # 浅紫色
leg2_exploded.paint_uniform_color(np.array([1.0, 0.7, 0.8]))  # 浅粉色
leg3_exploded.paint_uniform_color(np.array([0.7, 1.0, 0.8]))  # 浅绿色
leg4_exploded.paint_uniform_color(np.array([0.8, 0.8, 0.7]))  # 浅黄色

# 创建可视化窗口
vis2 = o3d.visualization.Visualizer()
vis2.create_window()

# 添加所有部件到可视化器
vis2.add_geometry(table_top_exploded)
vis2.add_geometry(leg1_exploded)
vis2.add_geometry(leg2_exploded)
vis2.add_geometry(leg3_exploded)
vis2.add_geometry(leg4_exploded)

# 设置渲染选项
opt2 = vis2.get_render_option()
opt2.background_color = np.asarray([1, 1, 1])  # 白色背景
opt2.point_size = 1.0
opt2.line_width = 1.0  # 设置线宽
opt2.mesh_show_wireframe = True  # 显示线框
opt2.mesh_show_back_face = True  # 显示背面

# 运行可视化
vis2.run()
vis2.destroy_window()

# 加载兔子模型
gt_mesh = o3d.data.BunnyMesh()
gt_mesh = o3d.io.read_triangle_mesh(gt_mesh.path)
gt_mesh.compute_vertex_normals()

# 设置浅褐色
light_brown = np.array([0.8, 0.7, 0.6])  # RGB值，范围0-1
gt_mesh.paint_uniform_color(light_brown)

# 使用Poisson Disk采样
pcd = gt_mesh.sample_points_poisson_disk(3000)

# 计算法向量
pcd.estimate_normals()

# 使用Ball Pivoting算法重建网格
radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
rec_mesh.compute_vertex_normals()

# 创建可视化窗口
vis3 = o3d.visualization.Visualizer()
vis3.create_window()

# 添加点云和重建的网格到可视化器
# vis3.add_geometry(pcd)
vis3.add_geometry(gt_mesh)

# 设置渲染选项
opt3 = vis3.get_render_option()
opt3.background_color = np.asarray([1, 1, 1])  # 白色背景
opt3.point_size = 2.0  # 设置点的大小
opt3.line_width = 1.0  # 设置线宽
# opt3.mesh_show_wireframe = True  # 显示线框
opt3.mesh_show_back_face = True  # 显示背面

# 运行可视化
vis3.run()
vis3.destroy_window()
