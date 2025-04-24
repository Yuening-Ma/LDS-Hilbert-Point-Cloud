import os
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from adjustText import adjust_text 
from hilbert import decode
from mpl_toolkits.mplot3d import Axes3D


def generate_hilbert_curve_3d(order):

    # 定义缓存路径
    cache_dir = "dim_3/.hilbert_cache"
    cache_file = os.path.join(cache_dir, f"hilbert_order_{order}.npy")

    # 检查缓存是否存在
    if os.path.exists(cache_file):
        # print(f"Loading Hilbert curve from cache: {cache_file}")
        curve = np.load(cache_file)
    else:
        # print(f"Generating Hilbert curve for order {order} and saving to cache...")
        # 如果缓存不存在，正常计算
        n = 8**order
        curve = decode(np.arange(n), 3, order)
        curve = (2*curve + 1) / (2**(order+1))
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存到缓存
        np.save(cache_file, curve)

    return curve


def create_hilbert_tree(hilbert_points):
    """ 创建一个 cKDTree，用于快速查找最近的希尔伯特点。 """
    return cKDTree(hilbert_points)

def closest_point_on_segment(point, segment_start, segment_end):
    """ 找到线段上离给定点最近的点。 """
    seg_vec = segment_end - segment_start
    point_vec = point - segment_start
    proj_factor = np.dot(point_vec, seg_vec) / np.dot(seg_vec, seg_vec)
    proj_factor = np.clip(proj_factor, 0, 1)  # 限制在 [0, 1] 范围内
    closest_point = segment_start + proj_factor * seg_vec
    return closest_point, proj_factor

def map_points_to_hilbert(points, hilbert_tree, hilbert_points):
    """ 将点云映射到最近的希尔伯特曲线段上。 """
    mapped_points = []
    segment_indices = []
    segment_positions = []
    for point in points:
        # 找到最近的希尔伯特点的索引
        _, nearest_index = hilbert_tree.query(point)
        
        # 获取最近点及其邻居点
        if nearest_index == 0:
            segment_start = hilbert_points[nearest_index]
            segment_end = hilbert_points[nearest_index + 1]
            segment_index = nearest_index
            _, proj_factor = closest_point_on_segment(point, segment_start, segment_end)
        elif nearest_index == len(hilbert_points) - 1:
            segment_start = hilbert_points[nearest_index - 1]
            segment_end = hilbert_points[nearest_index]
            segment_index = nearest_index - 1
            _, proj_factor = closest_point_on_segment(point, segment_start, segment_end)
        else:
            segment_start1 = hilbert_points[nearest_index - 1]
            segment_end1 = hilbert_points[nearest_index]
            segment_start2 = hilbert_points[nearest_index]
            segment_end2 = hilbert_points[nearest_index + 1]
            
            # 计算两个线段上的最近点
            closest_point1, proj_factor1 = closest_point_on_segment(point, segment_start1, segment_end1)
            closest_point2, proj_factor2 = closest_point_on_segment(point, segment_start2, segment_end2)
            
            # 选择更近的线段
            if np.linalg.norm(point - closest_point1) < np.linalg.norm(point - closest_point2):
                segment_start, segment_end = segment_start1, segment_end1
                segment_index = nearest_index - 1
                proj_factor = proj_factor1
            else:
                segment_start, segment_end = segment_start2, segment_end2
                segment_index = nearest_index
                proj_factor = proj_factor2
        
        # 找到选定线段上的最近点
        closest_point = segment_start + proj_factor * (segment_end - segment_start)
        mapped_points.append(closest_point)
        segment_indices.append(segment_index)
        segment_positions.append(proj_factor)

    return np.array(mapped_points), np.array(segment_indices), np.array(segment_positions)

def sort_hilbert(pc, order=3):
    """ 将点云按希尔伯特曲线排序。 """
    hilbert_points = generate_hilbert_curve_3d(order)
    hilbert_tree = create_hilbert_tree(hilbert_points)

    mapped_cloud, segment_indices, segment_positions = map_points_to_hilbert(pc, hilbert_tree, hilbert_points)

    # 将点云按希尔伯特曲线段索引和投影位置排序
    combined = np.column_stack((segment_indices, segment_positions))
    sorted_indices = np.lexsort((combined[:, 1], combined[:, 0]))

    return sorted_indices, mapped_cloud, hilbert_points


# 测试代码
n = 100
order = 2
# point_cloud = np.random.rand(n, 3)
# np.savetxt('code_for_essay/data/dim3_pc_N100.csv', point_cloud, delimiter=',')
point_cloud = np.loadtxt('code_for_essay/data/dim3_pc_N100.csv', delimiter=',')

start_time = time.time()
sorted_indices, mapped_cloud, hilbert_points = sort_hilbert(point_cloud, order=order)
print(f"排序耗时: {time.time() - start_time:.4f} 秒")

# 使用PCA计算主方向
pca = PCA(n_components=3)
pca.fit(point_cloud)
principal_direction = pca.components_[0]  # 第一个主成分方向

# 按主方向投影并排序
projections = np.dot(point_cloud - [0.5, 0.5, 0.5], principal_direction)  # 相对于 (0.5, 0.5, 0.5) 的投影
pca_sorted_indices = np.argsort(projections)

# 计算投影点云（投影到经过 (0.5, 0.5, 0.5) 的主方向上的点）
pca_projected_points = np.outer(projections, principal_direction) + [0.5, 0.5, 0.5]

# 创建1行2列的子图
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# 第一个subplot：PCA排序
ax1.set_title('PCA Sorted Point Cloud')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)
ax1.set_box_aspect([1, 1, 1])  # x:y:z 的比例

# 绘制原始点云
ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', label='Original Point Cloud', s=40)

# 绘制投影点云（投影到主方向上的点）
ax1.scatter(pca_projected_points[:, 0], pca_projected_points[:, 1], pca_projected_points[:, 2], c='g', marker='x', label='Projected Points', s=40)

# 连接原始点和投影点
for i in range(n):
    ax1.plot([point_cloud[i, 0], pca_projected_points[i, 0]],
             [point_cloud[i, 1], pca_projected_points[i, 1]],
             [point_cloud[i, 2], pca_projected_points[i, 2]], c='gray', linestyle='--', linewidth=1)

# 标注投影点的序号（标注在原始点旁边）
for i, idx in enumerate(pca_sorted_indices):
    ax1.text(point_cloud[idx, 0], point_cloud[idx, 1], point_cloud[idx, 2], str(i), fontsize=12, ha='right', va='bottom')

# 绘制主成分方向（长箭头）
ax1.quiver(0.5, 0.5, 0.5, principal_direction[0], principal_direction[1], principal_direction[2], color='r', label='Principal Direction')

# 第二个subplot：希尔伯特排序
ax2.set_title(f'Point Cloud Mapped to Hilbert Curve (Order {order})')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0, 1)
ax2.set_box_aspect([1, 1, 1])  # x:y:z 的比例

# 绘制希尔伯特曲线
for i in range(len(hilbert_points) - 1):
    start_point = hilbert_points[i]
    end_point = hilbert_points[i + 1]
    ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], 
             color='r', label=f'Segment {i}' if i == 0 else "")

# 绘制原始点云
ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', label='Original Point Cloud', s=40)

# 绘制投影后的点云
ax2.scatter(mapped_cloud[:, 0], mapped_cloud[:, 1], mapped_cloud[:, 2], c='g', marker='x', label='Mapped Points', s=40)

# 连接原始点和投影点
for i in range(n):
    ax2.plot([point_cloud[i, 0], mapped_cloud[i, 0]],
             [point_cloud[i, 1], mapped_cloud[i, 1]],
             [point_cloud[i, 2], mapped_cloud[i, 2]], c='gray', linestyle='--', linewidth=1)

# 标注投影点的序号（标注在原始点旁边）
for i, idx in enumerate(sorted_indices):
    ax2.text(point_cloud[idx, 0], point_cloud[idx, 1], point_cloud[idx, 2], str(i), fontsize=12, ha='right', va='bottom')

# 添加图例
ax1.legend(loc='upper right', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)

# 调整子图间距
plt.tight_layout(pad=1.0)  # 设置四边距为 5.0
plt.subplots_adjust(hspace=0.2, wspace=00.1)  # 调整子图之间的行间距和列间距

# 保存和显示图像
plt.savefig(f'code_for_essay/output/method_2.png', dpi=300)
plt.savefig(f'code_for_essay/output/method_2.svg', format='svg')

plt.show()