import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 模型名称列表
model_names = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]

# 创建一个figure和4行3列的子图
fig, axs = plt.subplots(4, 3, figsize=(14, 18), subplot_kw={'projection': '3d'})

# 定义一个函数用于绘制mesh
def plot_mesh(ax, mesh, title, edgecolor='none'):
    # 获取mesh的顶点和三角形
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # 绘制mesh
    collection = Poly3DCollection(vertices[triangles])
    collection.set_edgecolor(edgecolor)  # 设置边的颜色为黑色
    collection.set_facecolor([0.8, 0.9, 1.0])  # 设置面的颜色为浅蓝色
    collection.set_linewidth(0.05)  # 设置边线的粗细为0.5
    ax.add_collection3d(collection)
    
    # 设置坐标轴范围
    ax.set_xlim(np.min(vertices[:, 0]), np.max(vertices[:, 0]))
    ax.set_ylim(np.min(vertices[:, 1]), np.max(vertices[:, 1]))
    ax.set_zlim(np.min(vertices[:, 2]), np.max(vertices[:, 2]))
    
    # 设置标题
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=135)  # 设置视角

# 遍历模型名称列表
for i, model_name in enumerate(model_names):
    # 计算当前模型在子图中的行和列索引
    row = (i // 3) * 2  # 每3个模型换一行
    col = i % 3  # 每行3个模型
    
    # 读取OFF文件为mesh，并绘制到第一行的子图
    off_mesh = o3d.io.read_triangle_mesh(f"datasets/ModelNet40/off/{model_name}_0001.off")
    plot_mesh(axs[row, col], off_mesh, f"OFF_{model_name}", edgecolor='black')
    
    # 读取PLY文件为mesh，并绘制到第二行的子图
    ply_mesh = o3d.io.read_triangle_mesh(f"datasets/ModelNet40/ply_alpha/{model_name}_0001.ply", edgecolor='black')
    plot_mesh(axs[row + 1, col], ply_mesh, f"PLY_{model_name}")

# 调整子图间距
plt.tight_layout(pad=5.0)  # 设置四边距为 5.0
plt.subplots_adjust(hspace=0.2, wspace=0.2)  # 调整子图之间的行间距和列间距

# 保存和显示图像
plt.savefig(f'code_for_essay/output/exp4_dim3_mesh.png', dpi=300)
# plt.savefig(f'code_for_essay/output/exp4_dim3_mesh.svg', format='svg')
plt.show()