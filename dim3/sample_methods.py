import numpy as np
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
from sklearn.kernel_approximation import Nystroem
from sklearn import preprocessing
import fpsample
from scipy.spatial import cKDTree
import os
import open3d as o3d
from hilbert import decode


def normalization(pc):

    min_max_scaler = preprocessing.MinMaxScaler()
    pc_normal = min_max_scaler.fit_transform(pc)

    return pc_normal


def normalization_keep_ratio(pc):

    pc_xyz = pc[:, :3]

    maxs = np.max(pc_xyz, axis=0)
    mins = np.min(pc_xyz, axis=0)

    extention = maxs - mins
    max_extention = np.max(extention)

    pc_xyz_normal = (pc_xyz - mins) / max_extention

    pc_feature = pc[:, 3:]
    min_max_scaler = preprocessing.MinMaxScaler()
    pc_feature_normal = min_max_scaler.fit_transform(pc_feature)

    pc_normal = np.hstack((pc_xyz_normal, pc_feature_normal))

    return pc_normal


def sort_PCA(pc, dim_reduction='PCA'):

    n, s = pc.shape

    if dim_reduction == 'PCA':
        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc)[:,0]

    elif dim_reduction == 'KernelPCA-linear':
        nys = Nystroem(kernel='linear', n_components=s)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]

    elif dim_reduction == 'KernelPCA-poly':
        nys = Nystroem(kernel='poly', n_components=s)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]

    elif dim_reduction == 'KernelPCA-rbf':
        nys = Nystroem(kernel='rbf', n_components=s)
        pc_nys = nys.fit_transform(pc)

        pca = PCA(n_components = 1)
        pc_principle = pca.fit_transform(pc_nys)[:,0]

    elif dim_reduction == 'KernelPCA-sigmoid':
        nys = Nystroem(kernel='sigmoid', n_components=s)
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

def closest_point_on_segment_batch(points, segment_starts, segment_ends):
    """ 批量计算点到线段的投影 """
    # 计算线段向量
    seg_vecs = segment_ends - segment_starts  # shape: (n, 3)
    
    # 点到线段起点的向量
    point_vecs = points - segment_starts  # shape: (n, 3)
    
    # 计算投影因子
    # 对于每行，计算点积
    numerators = np.sum(point_vecs * seg_vecs, axis=1)  # shape: (n,)
    denominators = np.sum(seg_vecs * seg_vecs, axis=1)  # shape: (n,)
    proj_factors = numerators / denominators  # shape: (n,)
    
    # 限制投影因子在 [0, 1] 范围内
    proj_factors = np.clip(proj_factors, 0, 1)  # shape: (n,)
    
    # 计算最近点
    closest_points = segment_starts + np.outer(proj_factors, np.ones(3)) * seg_vecs  # shape: (n, 3)
    
    return closest_points, proj_factors

def map_points_to_hilbert(points, hilbert_tree, hilbert_points):
    """ 将点云映射到最近的希尔伯特曲线段上。 """
    # 找到每个点最近的希尔伯特点的索引
    _, nearest_indices = hilbert_tree.query(points)
    
    n_points = len(points)
    hp_len = len(hilbert_points)
    
    # 处理边界情况：第一个和最后一个点
    is_first = (nearest_indices == 0)
    is_last = (nearest_indices == hp_len - 1)
    is_middle = ~(is_first | is_last)
    
    # 预分配结果数组
    segment_indices = np.zeros(n_points, dtype=int)
    segment_positions = np.zeros(n_points)
    mapped_points = np.zeros((n_points, points.shape[1]))
    
    # 处理第一个点
    if np.any(is_first):
        first_points = points[is_first]
        first_indices = nearest_indices[is_first]
        
        segment_starts = hilbert_points[first_indices]
        segment_ends = hilbert_points[first_indices + 1]
        
        closest_points, proj_factors = closest_point_on_segment_batch(first_points, segment_starts, segment_ends)
        
        segment_indices[is_first] = first_indices
        segment_positions[is_first] = proj_factors
        mapped_points[is_first] = closest_points
    
    # 处理最后一个点
    if np.any(is_last):
        last_points = points[is_last]
        last_indices = nearest_indices[is_last]
        
        segment_starts = hilbert_points[last_indices - 1]
        segment_ends = hilbert_points[last_indices]
        
        closest_points, proj_factors = closest_point_on_segment_batch(last_points, segment_starts, segment_ends)
        
        segment_indices[is_last] = last_indices - 1
        segment_positions[is_last] = proj_factors
        mapped_points[is_last] = closest_points
    
    # 处理中间的点
    if np.any(is_middle):
        middle_points = points[is_middle]
        middle_indices = nearest_indices[is_middle]
        
        # 为每个点获取两个可能的线段
        segment_starts1 = hilbert_points[middle_indices - 1]
        segment_ends1 = hilbert_points[middle_indices]
        segment_starts2 = hilbert_points[middle_indices]
        segment_ends2 = hilbert_points[middle_indices + 1]
        
        # 计算两个线段上的最近点
        closest_points1, proj_factors1 = closest_point_on_segment_batch(middle_points, segment_starts1, segment_ends1)
        closest_points2, proj_factors2 = closest_point_on_segment_batch(middle_points, segment_starts2, segment_ends2)
        
        # 计算点到两个最近点的距离
        dist1 = np.sum((middle_points - closest_points1) ** 2, axis=1)
        dist2 = np.sum((middle_points - closest_points2) ** 2, axis=1)
        
        # 选择更近的线段
        closer_to_first = dist1 < dist2
        
        # 为每个中间点分配正确的线段索引和投影位置
        temp_segment_indices = np.where(closer_to_first, middle_indices - 1, middle_indices)
        temp_proj_factors = np.where(closer_to_first, proj_factors1, proj_factors2)
        temp_closest_points = np.where(closer_to_first[:, np.newaxis], closest_points1, closest_points2)
        
        segment_indices[is_middle] = temp_segment_indices
        segment_positions[is_middle] = temp_proj_factors
        mapped_points[is_middle] = temp_closest_points
    
    return mapped_points, segment_indices, segment_positions

def sort_hilbert(pc, order=3):
    """ 将点云按希尔伯特曲线排序。 """
    hilbert_points = generate_hilbert_curve_3d(order)
    hilbert_tree = create_hilbert_tree(hilbert_points)

    mapped_cloud, segment_indices, segment_positions = map_points_to_hilbert(pc, hilbert_tree, hilbert_points)

    # 将点云按希尔伯特曲线段索引和投影位置排序
    combined = np.column_stack((segment_indices, segment_positions))
    sorted_indices = np.lexsort((combined[:, 1], combined[:, 0]))

    return sorted_indices, mapped_cloud, hilbert_points


def random_sampling_cluster(pc, num_cluster, random_seed=5):
    np.random.seed(random_seed)
    n = pc.shape[0]
    M, s = divmod(n, num_cluster)  # 计算整除结果 M 和余数 s
    column_sizes = [M + 1] * s + [M] * (num_cluster - s)
    
    # 初始化输出数组，每个群占用三列
    if s == 0:
        pc_sampled = np.zeros((M, num_cluster * 3))  # 每个群三列
    else:
        pc_sampled = np.zeros((M + 1, num_cluster * 3))  # 每个群三列

    choices = np.random.choice(n, n, replace=False)  # 随机打乱索引

    start = 0
    for i in range(num_cluster):
        size = column_sizes[i]
        end = start + size
        # 每个群的 x 和 y 坐标分别放在三列
        pc_sampled[:size, i * 3] = pc[choices[start:end], 0]  # x 坐标
        pc_sampled[:size, i * 3 + 1] = pc[choices[start:end], 1]  # y 坐标
        pc_sampled[:size, i * 3 + 2] = pc[choices[start:end], 2]  # z 坐标
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
        hilbert_order = max(round(np.log(n) / np.log(8)), 1) if order is None else order
        sorted_indices, _, _ = sort_hilbert(pc_normal, hilbert_order)
    else:
        raise ValueError("sort for bds_sampling_cluster must be 'PCA' or 'Hilbert'!")
    
    pc_sorted = pc[sorted_indices]
    
    # 生成BDS序列
    array = np.arange(1, n+1) * trans_num % 1
    temp = np.argsort(array)
    ranks = np.argsort(temp)

    if s == 0:
        pc_sampled = np.zeros((M, num_cluster * 3))  # 每个群三列
    else:
        pc_sampled = np.zeros((M + 1, num_cluster * 3))  # 每个群三列

    start = 0
    for i in range(num_cluster):
        size = column_sizes[i]
        end = start + size
        # 每个群的 x 和 y 坐标分别放在三列
        pc_sampled[:size, i * 3] = pc_sorted[ranks[start:end], 0]  # x 坐标
        pc_sampled[:size, i * 3 + 1] = pc_sorted[ranks[start:end], 1]  # y 坐标
        pc_sampled[:size, i * 3 + 2] = pc_sorted[ranks[start:end], 2]  # z 坐标
        start = end

    return pc_sampled


def bds_sampling(pc, num_sample, trans_num=np.e, sort='PCA', dim_reduction='PCA', order=None):

    n = pc.shape[0]

    pc_normal = normalization(pc)
    if sort == 'PCA':
        sorted_indices = sort_PCA(pc_normal, dim_reduction)
    elif sort == 'Hilbert':
        hilbert_order = max(round(np.log(n) / np.log(8)), 1) if order is None else order
        sorted_indices, _, _ = sort_hilbert(pc_normal, hilbert_order)
    else:
        raise ValueError("sort for bds_sampling_cluster must be 'PCA' or 'Hilbert'!")
    
    pc_sorted = pc[sorted_indices]
    
    # 生成BDS序列
    array = np.arange(1, n+1) * trans_num % 1
    temp = np.argsort(array)
    ranks = np.argsort(temp)

    pc_sampled = pc_sorted[ranks[:num_sample]]

    return pc_sampled


def fps(pc, num_sample, random_seed=5):

    np.random.seed(random_seed)
    pc_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, num_sample)

    return pc[pc_samples_idx]


def voxel_sampling(pc, voxel_size):

    # 将 NumPy 数组转换为 Open3D 点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    
    # 使用 Open3D 的体素降采样方法
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # [^2^]
    
    # 将降采样后的点云转换回 NumPy 数组
    downsampled_pc = np.asarray(downsampled_pcd.points)
    
    return downsampled_pc


def voxel_sampling_flexible(pc, num_sample, tol=0.10, max_iter=10):
    """
    动态调整体素大小以实现目标采样数量的体素降采样。
    
    Params:
        pc (np.ndarray): 输入的三维点云，形状为 (n, 3)。
        num_sample (int): 目标采样数量。
        tol (float): 目标数量的容忍范围（默认为 0.10，即 90% 到 110%）。
        max_iter (int): 最大迭代次数，防止无限循环。
    
    Return:
        np.ndarray: 降采样后的点云，形状为 (m, 3)。
    """
    # 初始估计 voxel_size
    scale_x = pc[:, 0].ptp()  # x 方向的范围（最大值 - 最小值）
    scale_y = pc[:, 1].ptp()  # y 方向的范围
    scale_z = pc[:, 2].ptp()  # z 方向的范围
    scale = max(scale_x, scale_y, scale_z)  # 选择最大的范围作为参考
    voxel_size = scale / np.cbrt(num_sample) * 0.5  # 初始估计的体素大小，假设每个体素产生 1 个点

    lower_bound = num_sample * (1 - tol)
    upper_bound = num_sample * (1 + tol)

    sampled_points = voxel_sampling(pc, voxel_size)  # 调用三维体素采样函数
    num_sampled = len(sampled_points)

    if num_sampled >= lower_bound and num_sampled <= upper_bound:
        return sampled_points
    else:
        voxel_size = voxel_size * np.cbrt(num_sampled / num_sample)

    ratio = 0.9  # 调整体素大小的比例因子
    iteration = 0

    while iteration < max_iter:
        sampled_points = voxel_sampling(pc, voxel_size)  # 调用三维体素采样函数
        num_sampled = len(sampled_points)

        # print(f'iteration: {iteration}, voxel_size: {voxel_size}, num_sampled: {num_sampled}')

        # 检查是否满足条件
        if (num_sampled >= lower_bound) and (num_sampled <= upper_bound):
            break

        # 调整 voxel_size
        if num_sampled < lower_bound:
            voxel_size *= ratio  # 减小 voxel_size
        else:
            voxel_size /= ratio  # 增大 voxel_size

        iteration += 1

    return sampled_points


if __name__ == '__main__':

    pass

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import time

    # # 测试代码
    # n = 100
    # order = 3
    # point_cloud = np.random.rand(n, 3)

    # start_time = time.time()
    # sorted_indices, mapped_cloud, hilbert_points = sort_hilbert(point_cloud, order=order)
    # print(f"排序耗时: {time.time() - start_time:.4f} 秒")

    # # 可视化
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制希尔伯特曲线
    # for i in range(len(hilbert_points) - 1):
    #     start_point = hilbert_points[i]
    #     end_point = hilbert_points[i + 1]
    #     ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], 
    #             color='r', label=f'Segment {i}' if i == 0 else "")

    # # 绘制原始点云
    # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', label='Original Point Cloud', s=25)

    # # 绘制投影后的点云
    # ax.scatter(mapped_cloud[:, 0], mapped_cloud[:, 1], mapped_cloud[:, 2], c='g', marker='x', label='Mapped Points', s=25)

    # # 标注投影点的序号
    # for i, idx in enumerate(sorted_indices):
    #     ax.text(mapped_cloud[idx, 0], mapped_cloud[idx, 1], mapped_cloud[idx, 2], str(i), fontsize=8, ha='right', va='bottom')

    # ax.set_title(f'Point Cloud Mapped to Hilbert Curve Segments (Order {order})')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    # ax.set_box_aspect([1, 1, 1])  # x:y:z 的比例
    # ax.legend()
    # plt.show()