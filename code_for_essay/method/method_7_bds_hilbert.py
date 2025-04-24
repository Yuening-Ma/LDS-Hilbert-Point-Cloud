import os
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
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

    n = 1230
    sampling_ratio = 0.1
    order = round(np.log(n) / np.log(4))

    pc_gt = np.loadtxt(f'code_for_essay/output/method/pc_gt_{n}.csv', delimiter=',')
    pc = np.loadtxt(f'code_for_essay/output/method/pc_{n}.csv', delimiter=',')

    print(pc_gt.shape, pc.shape)

    # 把pc标准化到(0,1)
    pc_normal = (pc - np.min(pc, axis=0)) / (np.max(pc, axis=0) - np.min(pc, axis=0))

    # 希尔伯特排序
    sorted_indices, mapped_cloud, hilbert_points = sort_hilbert(pc_normal, order=order)

    # 创建2行2列的子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # 第二个subplot：希尔伯特排序
    axs[0].set_title(f'Hilbert-Curve Sorted Point Cloud', fontsize=12, pad=20)
    axs[0].set_xlabel('X', fontsize=10)
    axs[0].set_ylabel('Y', fontsize=10)
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)

    # 绘制希尔伯特曲线
    for i in range(len(hilbert_points) - 1):
        start_point = hilbert_points[i]
        end_point = hilbert_points[i + 1]
        axs[0].plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color='r', label=f'Hilbert Segment' if i == 0 else "")

    # 绘制原始点云
    axs[0].scatter(pc_normal[:, 0], pc_normal[:, 1], c='b', marker='o', label='Original Point Cloud', s=5)

    # 绘制投影后的点云
    axs[0].scatter(mapped_cloud[:, 0], mapped_cloud[:, 1], c='g', marker='x', label='Mapped Points', s=20)

    # 连接原始点和投影点
    for i in range(n):
        axs[0].plot([pc_normal[i, 0], mapped_cloud[i, 0]],
                    [pc_normal[i, 1], mapped_cloud[i, 1]], c='gray', linestyle='--', linewidth=1)

    # # 标注投影点的序号（标注在原始点旁边）
    # for i, idx in enumerate(sorted_indices):
    #     axs[1].text(pc[idx, 0]+0.04, pc[idx, 1]-0.03, str(i), fontsize=12, ha='right', va='bottom')

    # 生成BDS序列
    array = np.arange(1, n+1) * np.e % 1
    temp = np.argsort(array)
    ranks = np.argsort(temp)

    # 第四个子图：BDS-Hilbert降采样点云
    pc_sorted = pc[sorted_indices]
    pc_sampled = pc_sorted[ranks[:int(n*sampling_ratio)]]

    # 绘制降采样点云
    axs[1].scatter(pc_sampled[:, 0], pc_sampled[:, 1], c='g', label='Sampled Points', s=20)

    axs[1].set_title(f'LDS-Hilbert Sampled Point Cloud', fontsize=12, pad=20)
    axs[1].set_xlabel('X', fontsize=10)
    axs[1].set_ylabel('Y', fontsize=10)
    axs[1].set_xlim(-1.1, 1.1)
    axs[1].set_ylim(-1.1, 1.1)

    # 图例
    axs[0].legend(fontsize=10)
    axs[1].legend(fontsize=10)

    # 调整子图间距
    plt.tight_layout(pad=1.0)  # 设置四边距为 5.0
    plt.subplots_adjust(hspace=0.2, wspace=0.2)  # 调整子图之间的行间距和列间距

    # 保存和显示图像
    plt.savefig(f'code_for_essay/output/method/method_7_lds.png', dpi=300)
    plt.savefig(f'code_for_essay/output/method/method_7_lds.svg', format='svg')

    plt.show()