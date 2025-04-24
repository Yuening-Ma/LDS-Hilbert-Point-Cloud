import os
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from adjustText import adjust_text 


def d2xy(n, d):
    """ Converts d-index to xy-coordinates (center of each cell). """
    t = d
    x = y = 0
    s = 1
    while (s < n):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    # 调整坐标到格子的中心
    x = (x + 0.5) / n
    y = (y + 0.5) / n
    return x, y

def generate_hilbert_curve(order):
    # 定义缓存路径
    cache_dir = "dim_2/.hilbert_cache"
    cache_file = os.path.join(cache_dir, f"hilbert_order_{order}.npy")
    
    # 检查缓存是否存在
    if os.path.exists(cache_file):
        print(f"Loading Hilbert curve from cache: {cache_file}")
        hilbert_points = np.load(cache_file)
    else:
        print(f"Generating Hilbert curve for order {order} and saving to cache...")
        # 如果缓存不存在，正常计算
        n = 2 ** order
        hilbert_points = np.array([d2xy(n, d) for d in range(n * n)])
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存到缓存
        np.save(cache_file, hilbert_points)
    
    return hilbert_points


def create_hilbert_tree(hilbert_points):
    """ Create a cKDTree for the Hilbert curve points. """
    return cKDTree(hilbert_points)

def map_points_to_hilbert(points, hilbert_tree, hilbert_points):
    """ Maps points to the nearest Hilbert segment. """
    mapped_points = []
    segment_indices = []
    segment_positions = []
    for point in points:
        # Find the index of the nearest Hilbert point
        _, nearest_index = hilbert_tree.query(point)
        
        # Get the nearest point and its neighbors
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
            
            # Calculate the closest points on both segments
            closest_point1, proj_factor1 = closest_point_on_segment(point, segment_start1, segment_end1)
            closest_point2, proj_factor2 = closest_point_on_segment(point, segment_start2, segment_end2)
            
            # Choose the closer segment
            if np.linalg.norm(point - closest_point1) < np.linalg.norm(point - closest_point2):
                segment_start, segment_end = segment_start1, segment_end1
                segment_index = nearest_index - 1
                proj_factor = proj_factor1
            else:
                segment_start, segment_end = segment_start2, segment_end2
                segment_index = nearest_index
                proj_factor = proj_factor2
        
        # Find the closest point on the chosen segment
        closest_point = segment_start + proj_factor * (segment_end - segment_start)
        mapped_points.append(closest_point)
        segment_indices.append(segment_index)
        segment_positions.append(proj_factor)

    return np.array(mapped_points), np.array(segment_indices), np.array(segment_positions)

def closest_point_on_segment(point, segment_start, segment_end):
    """ Finds the closest point on a line segment to a given point. """
    seg_vec = segment_end - segment_start
    point_vec = point - segment_start
    proj_factor = np.dot(point_vec, seg_vec) / np.dot(seg_vec, seg_vec)
    proj_factor = np.clip(proj_factor, 0, 1)  # Clamp to segment bounds
    closest_point = segment_start + proj_factor * seg_vec
    return closest_point, proj_factor


def sort_hilbert(pc, order=4):

    hilbert_points = generate_hilbert_curve(order)
    hilbert_tree = create_hilbert_tree(hilbert_points)

    mapped_cloud, segment_indices, segment_positions = map_points_to_hilbert(pc, hilbert_tree, hilbert_points)

    combined = np.column_stack((segment_indices, segment_positions))
    sorted_indices = np.lexsort((combined[:, 1], combined[:, 0]))

    return sorted_indices, mapped_cloud, hilbert_points
    # return sorted_indices
    

if __name__ == '__main__':

    n = 100
    order = 3

    # point_cloud = np.random.rand(n, 2)
    # np.savetxt('code_for_essay/data/dim2_pc_N100.csv', point_cloud, delimiter=',')
    point_cloud = np.loadtxt('code_for_essay/data/dim2_pc_N100.csv', delimiter=',')

    start_time = time.time()
    sorted_indices, mapped_cloud, hilbert_points = sort_hilbert(point_cloud, order=order)
    print(time.time() - start_time)

    # 使用PCA计算主方向
    pca = PCA(n_components=2)
    pca.fit(point_cloud)
    principal_direction = pca.components_[0]  # 第一个主成分方向

    # 按主方向投影并排序
    projections = np.dot(point_cloud - [0.5, 0.5], principal_direction)  # 相对于 (0.5, 0.5) 的投影
    pca_sorted_indices = np.argsort(projections)

    # 计算投影点云（投影到经过 (0.5, 0.5) 的主方向上的点）
    pca_projected_points = np.outer(projections, principal_direction) + [0.5, 0.5]

    # 创建1行2列的子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))


    # 第一个subplot：PCA排序
    axs[0].set_title('PCA Sorted Point Cloud', fontsize=12, pad=20)
    axs[0].set_xlabel('X', fontsize=10)
    axs[0].set_ylabel('Y', fontsize=10)
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)

    # 绘制原始点云
    axs[0].scatter(point_cloud[:, 0], point_cloud[:, 1], c='b', marker='o', label='Original Point Cloud', s=40)

    # 绘制投影点云（投影到主方向上的点）
    axs[0].scatter(pca_projected_points[:, 0], pca_projected_points[:, 1], c='g', marker='x', label='Projected Points', s=40)

    # 连接原始点和投影点
    for i in range(n):
        axs[0].plot([point_cloud[i, 0], pca_projected_points[i, 0]],
                    [point_cloud[i, 1], pca_projected_points[i, 1]], c='gray', linestyle='--', linewidth=1)

    # 标注投影点的序号（标注在原始点旁边）
    for i, idx in enumerate(pca_sorted_indices):
        axs[0].text(point_cloud[idx, 0]+0.04, point_cloud[idx, 1]-0.03, str(i), fontsize=12, ha='right', va='bottom')

    # 绘制主成分方向（长箭头）
    axs[0].arrow(0.5, 0.5, principal_direction[0]*0.4, principal_direction[1]*0.4, head_width=0.05, head_length=0.05, fc='r', ec='r', label='Principal Direction')

    # 第二个subplot：希尔伯特排序
    axs[1].set_title(f'Hilbert-Curve Sorted Point Cloud', fontsize=12, pad=20)
    axs[1].set_xlabel('X', fontsize=10)
    axs[1].set_ylabel('Y', fontsize=10)
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)

    # 绘制希尔伯特曲线
    for i in range(len(hilbert_points) - 1):
        start_point = hilbert_points[i]
        end_point = hilbert_points[i + 1]
        axs[1].plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color='r', label=f'Segment {i}' if i == 0 else "")

    # 绘制原始点云
    axs[1].scatter(point_cloud[:, 0], point_cloud[:, 1], c='b', marker='o', label='Original Point Cloud', s=40)

    # 绘制投影后的点云
    axs[1].scatter(mapped_cloud[:, 0], mapped_cloud[:, 1], c='g', marker='x', label='Mapped Points', s=40)

    # 连接原始点和投影点
    for i in range(n):
        axs[1].plot([point_cloud[i, 0], mapped_cloud[i, 0]],
                    [point_cloud[i, 1], mapped_cloud[i, 1]], c='gray', linestyle='--', linewidth=1)

    # 标注投影点的序号（标注在原始点旁边）
    for i, idx in enumerate(sorted_indices):
        axs[1].text(point_cloud[idx, 0]+0.04, point_cloud[idx, 1]-0.03, str(i), fontsize=12, ha='right', va='bottom')

    # 将图例放到右上角
    axs[0].legend(loc='upper right', fontsize=10)
    axs[1].legend(loc='upper right', fontsize=10)

    # 调整子图间距
    plt.tight_layout(pad=1.0)  # 设置四边距为 5.0
    plt.subplots_adjust(hspace=0.2, wspace=0.2)  # 调整子图之间的行间距和列间距

    # 保存和显示图像
    plt.savefig(f'code_for_essay/output/method_1.png', dpi=300)
    plt.savefig(f'code_for_essay/output/method_1.svg', format='svg')

    plt.show()