import numpy as np
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
from sklearn.kernel_approximation import Nystroem
from sklearn import preprocessing
import fpsample
from scipy.spatial import cKDTree
import os


def normalization(pc):

    min_max_scaler = preprocessing.MinMaxScaler()
    pc_normal = min_max_scaler.fit_transform(pc)

    return pc_normal


def sort_PCA(pc, dim_reduction='PCA'):

    n = pc.shape[0] 

    if dim_reduction == 'PCA':
        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc)[:,0]

    elif dim_reduction == 'KernelPCA-linear':
        nys = Nystroem(kernel='linear', n_components=3)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]

    elif dim_reduction == 'KernelPCA-poly':
        nys = Nystroem(kernel='poly', n_components=3)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]

    elif dim_reduction == 'KernelPCA-rbf':
        nys = Nystroem(kernel='rbf', n_components=3)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]

    elif dim_reduction == 'KernelPCA-sigmoid':
        nys = Nystroem(kernel='sigmoid', n_components=3)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]
    
    elif dim_reduction == 'FA':
        fa = FactorAnalysis(n_components = 1)
        pc_principle = fa.fit_transform(pc)[:,0]

    elif dim_reduction == 'TruncatedSVD':
        tsvd = TruncatedSVD(n_components = 1)
        pc_principle = tsvd.fit_transform(pc)[:,0]

    else:
        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc)[:,0]

    indices_pc_principle = np.argsort(pc_principle)

    return indices_pc_principle


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

    # return sorted_indices, mapped_cloud, hilbert_points
    return sorted_indices


import numpy as np

def random_sampling_cluster(pc, num_cluster, random_seed=5):
    np.random.seed(random_seed)
    n = pc.shape[0]
    M, s = divmod(n, num_cluster)  # 计算整除结果 M 和余数 s
    column_sizes = [M + 1] * s + [M] * (num_cluster - s)
    
    # 初始化输出数组，每个群占用两列
    if s == 0:
        pc_sampled = np.zeros((M, num_cluster * 2))  # 每个群两列
    else:
        pc_sampled = np.zeros((M + 1, num_cluster * 2))  # 每个群两列

    choices = np.random.choice(n, n, replace=False)  # 随机打乱索引

    start = 0
    for i in range(num_cluster):
        size = column_sizes[i]
        end = start + size
        # 每个群的 x 和 y 坐标分别放在两列
        pc_sampled[:size, i * 2] = pc[choices[start:end], 0]  # x 坐标
        pc_sampled[:size, i * 2 + 1] = pc[choices[start:end], 1]  # y 坐标
        start = end

    return pc_sampled

# n = 22
# pc = np.vstack((np.arange(1,n+1), np.arange(2,n+2))).T
# pc_sampled = random_sampling_cluster(pc, 3)
# print(pc_sampled)


def bds_sampling_cluster(pc, num_cluster, trans_num=np.e, sort='PCA', dim_reduction='PCA', order=None):

    n = pc.shape[0]
    M, s = divmod(n, num_cluster)  # 计算整除结果 M 和余数 s
    column_sizes = [M + 1] * s + [M] * (num_cluster - s)

    pc_normal = normalization(pc)
    if sort == 'PCA':
        sorted_indices = sort_PCA(pc_normal, dim_reduction)
    elif sort == 'Hilbert':
        hilbert_order = round(np.log(n) / np.log(4)) if order is None else order
        sorted_indices = sort_hilbert(pc_normal, hilbert_order)
    else:
        raise ValueError("sort for bds_sampling_cluster must be 'PCA' or 'Hilbert'!")
    
    pc_sorted = pc[sorted_indices]
    
    # 生成BDS序列
    array = np.arange(1, n+1) * trans_num % 1
    temp = np.argsort(array)
    ranks = np.argsort(temp)

    if s == 0:
        pc_sampled = np.zeros((M, num_cluster * 2))  # 每个群两列
    else:
        pc_sampled = np.zeros((M + 1, num_cluster * 2))  # 每个群两列

    start = 0
    for i in range(num_cluster):
        size = column_sizes[i]
        end = start + size
        # 每个群的 x 和 y 坐标分别放在两列
        pc_sampled[:size, i * 2] = pc_sorted[ranks[start:end], 0]  # x 坐标
        pc_sampled[:size, i * 2 + 1] = pc_sorted[ranks[start:end], 1]  # y 坐标
        start = end

    return pc_sampled

# n = 22
# pc = np.vstack((np.arange(1,n+1), np.arange(2,n+2))).T
# pc_sampled = bds_sampling_cluster(pc, 3)
# print(pc_sampled)


def fps(pc, num_sample, random_seed=5):

    np.random.seed(random_seed)
    pc_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, num_sample)

    return pc[pc_samples_idx]


def voxel_sampling(pc, voxel_size):
    # 分别计算 x 和 y 维度的体素索引
    voxel_indices_x = np.floor((pc[:, 0] - pc[:, 0].min()) / voxel_size).astype(int)
    voxel_indices_y = np.floor((pc[:, 1] - pc[:, 1].min()) / voxel_size).astype(int)

    # 将二维索引合并为一个唯一索引
    unique_indices, inverse_indices = np.unique(
        np.vstack((voxel_indices_x, voxel_indices_y)).T, 
        axis=0, 
        return_inverse=True
    )

    # 计算每个体素单元中点的数量
    voxel_counts = np.bincount(inverse_indices)

    # 计算每个体素单元中点的 x 和 y 坐标的加权和
    voxel_sum_x = np.bincount(inverse_indices, weights=pc[:, 0])
    voxel_sum_y = np.bincount(inverse_indices, weights=pc[:, 1])

    # 计算每个体素单元的平均值
    voxel_means_x = voxel_sum_x / voxel_counts
    voxel_means_y = voxel_sum_y / voxel_counts

    # 将 x 和 y 坐标的平均值组合成二维点
    sampled_points = np.vstack((voxel_means_x, voxel_means_y)).T

    return sampled_points

# n = 1000
# pc = np.random.randn(n, 2)
# pc_sampled = voxel_sampling(pc, voxel_size=0.5)
# print(pc_sampled.shape)

# import matplotlib.pyplot as plt
# plt.scatter(pc_sampled[:, 0], pc_sampled[:, 1])
# # 设置网格线
# plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格线

# # 设置 x 轴和 y 轴的刻度间隔为 0.5
# plt.xticks([i * 0.5 for i in range(-5, 5)])  # 从 0 到 1.5，间隔为 0.5
# plt.yticks([i * 0.5 for i in range(-5, 5)])  # 同上
# plt.show()


def voxel_sampling_flexible(pc, num_sample, tol=0.10, max_iter=10):
    """
    动态调整体素大小以实现目标采样数量的体素降采样。
    
    Params:
        pc (np.ndarray): 输入的二维点云，形状为 (n, 2)。
        num_sample (int): 目标采样数量。
        tol (float): 目标数量的容忍范围（默认为 0.10，即 90% 到 110%）。
        max_iter (int): 最大迭代次数，防止无限循环。
    
    Return:
        np.ndarray: 降采样后的点云，形状为 (m, 2)。
    """
    # 初始估计 voxel_size
    scale_x = pc[:, 0].ptp()  # x 方向的范围（最大值 - 最小值）
    scale_y = pc[:, 1].ptp()  # y 方向的范围（最大值 - 最小值）
    scale = max(scale_x, scale_y)  # 选择较大的范围作为参考
    voxel_size = scale / np.sqrt(num_sample)  # 初始估计的体素大小，假设每个体素产生 1 个点

    lower_bound = num_sample * (1 - tol)
    upper_bound = num_sample * (1 + tol)

    sampled_points = voxel_sampling(pc, voxel_size)  # 调用二维体素采样函数
    num_sampled = len(sampled_points)

    if lower_bound <= num_sampled <= upper_bound:
        return sampled_points
    else:
        voxel_size = voxel_size * np.sqrt(num_sampled / num_sample)

    ratio = 0.9  # 调整体素大小的比例因子
    iteration = 0

    while iteration < max_iter:
        sampled_points = voxel_sampling(pc, voxel_size)  # 调用二维体素采样函数
        num_sampled = len(sampled_points)

        # 检查是否满足条件
        if lower_bound <= num_sampled <= upper_bound:
            break

        # 调整 voxel_size
        if num_sampled < lower_bound:
            voxel_size *= ratio  # 增大 voxel_size
        else:
            voxel_size /= ratio  # 减小 voxel_size

        iteration += 1

    return sampled_points

# n = 1000
# pc = np.random.randn(n, 2)
# pc_sampled = voxel_sampling_flexible(pc, num_sample=200)
# print(pc_sampled.shape)

# import matplotlib.pyplot as plt
# plt.scatter(pc_sampled[:, 0], pc_sampled[:, 1])
# # 设置网格线
# plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格线

# # 设置 x 轴和 y 轴的刻度间隔为 0.5
# plt.xticks([i * 0.5 for i in range(-5, 5)])  # 从 0 到 1.5，间隔为 0.5
# plt.yticks([i * 0.5 for i in range(-5, 5)])  # 同上
# plt.show()


if __name__ == '__main__':

    pass

    import matplotlib.pyplot as plt
    import time

    n = 1000
    order = 2
    point_cloud = np.random.rand(n, 2)

    start_time = time.time()
    sorted_indices, mapped_cloud, hilbert_points = sort_hilbert(point_cloud, order=order)
    print(time.time() - start_time)
    
    # Plotting
    plt.figure(figsize=(10, 10))

    # Generate a color map for segments
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(hilbert_points) - 1))

    # Plot each segment with a different color
    for i in range(len(hilbert_points) - 1):
        start_point = hilbert_points[i]
        end_point = hilbert_points[i + 1]
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                 color=colors[i], label=f'Segment {i}' if i == 0 else "")

    # Plot the original point cloud
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='b', marker='o', label='Point Cloud', s=25)

    # Plot the mapped points
    plt.scatter(mapped_cloud[:, 0], mapped_cloud[:, 1], c='g', marker='x', label='Mapped Points', s=25)

    # Annotate each mapped point with its sorted index
    for i, idx in enumerate(sorted_indices):
        plt.text(mapped_cloud[idx, 0]-0.005, mapped_cloud[idx, 1]+0.005, str(i), fontsize=8, ha='right', va='bottom')

    plt.title(f'Point Cloud Mapped to Hilbert Curve Segments (Order {order})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()