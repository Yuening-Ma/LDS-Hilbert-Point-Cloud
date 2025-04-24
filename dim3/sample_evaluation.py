
from scipy import stats
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import least_squares
import numpy.polynomial.polynomial as poly
import open3d as o3d
import trimesh
import pclpy
import os
import numpy as np


'''
距离
'''
def cloud_to_mesh(pc, mesh_file_path):

    mesh = trimesh.load(mesh_file_path)
    (closest_points, distances, triangle_id) = mesh.nearest.on_surface(pc[:, :3])

    return distances

def cloud_to_cloud_distance(pcd_1, pcd_2):

    dists = pcd_1.compute_point_cloud_distance(pcd_2)
    dists = np.asarray(dists)

    return np.mean(dists)

def hausdorff_distance(pc_1, pc_2, seed=0):
    # 度量了两个点集间的最大不匹配程度
    d1 = directed_hausdorff(pc_1, pc_2, seed=seed)[0]
    d2 = directed_hausdorff(pc_2, pc_1, seed=seed)[0]
    return max(d1, d2)

def asymmetric_hausdorff_distance(pc_1, pc_2, seed=0):
    # 度量了两个点集间的最大不匹配程度
    d1 = directed_hausdorff(pc_1, pc_2, seed=seed)[0]
    return d1

def hausdorff_distance_95(pc_1, pc_2):

    distances1 = cdist(pc_1, pc_2)
    distances2 = cdist(pc_2, pc_1)

    min_distances1 = np.min(distances1, axis=1)
    min_distances2 = np.min(distances2, axis=1)

    hd95_1 = np.percentile(min_distances1, 95)
    hd95_2 = np.percentile(min_distances2, 95)

    return max(hd95_1, hd95_2)

def asymmetric_hausdorff_distance_95(pc_1, pc_2):

    distances1 = cdist(pc_1, pc_2)
    min_distances1 = np.min(distances1, axis=1)
    hd95_1 = np.percentile(min_distances1, 95)
    return hd95_1

def chamfer_distance(pc_1, pc_2):
    """
    计算两个三维点云之间的Chamfer距离。
    """
    # 计算pc_1到pc_2的距离
    distances1 = cdist(pc_1, pc_2)
    min_distances1 = np.min(distances1, axis=1)
    chamfer_dist_1 = np.mean(min_distances1)

    # 计算pc_2到pc_1的距离
    distances2 = cdist(pc_2, pc_1)
    min_distances2 = np.min(distances2, axis=1)
    chamfer_dist_2 = np.mean(min_distances2)

    return chamfer_dist_1 + chamfer_dist_2


def pcd_distance(pcd_1, pcd_2):
    """
    计算两个点云之间的多种距离指标。
    
    参数:
        pcd_1, pcd_2: open3d.geometry.PointCloud 对象
    
    返回:
        list: 包含以下指标的列表：
              - asymmetric_hausdorff_1
              - asymmetric_hausdorff_2
              - asymmetric_hausdorff_95_1
              - asymmetric_hausdorff_95_2
              - cloud_to_cloud_distance_1
              - cloud_to_cloud_distance_2
              - chamfer_distance
    """
    # 获取点云之间的距离
    dists_1 = np.asarray(pcd_1.compute_point_cloud_distance(pcd_2))
    dists_2 = np.asarray(pcd_2.compute_point_cloud_distance(pcd_1))
    
    # 计算不对称Hausdorff距离
    asymmetric_hausdorff_1 = np.max(dists_1)
    asymmetric_hausdorff_2 = np.max(dists_2)
    
    # 计算不对称Hausdorff 95%距离
    asymmetric_hausdorff_95_1 = np.percentile(dists_1, 95)
    asymmetric_hausdorff_95_2 = np.percentile(dists_2, 95)
    
    # 计算点云到点云的平均距离
    cloud_to_cloud_distance_1 = np.mean(dists_1)
    cloud_to_cloud_distance_2 = np.mean(dists_2)
    
    # 计算Chamfer距离
    chamfer_distance = cloud_to_cloud_distance_1 + cloud_to_cloud_distance_2
    
    # 返回所有指标
    return [
        asymmetric_hausdorff_1,
        asymmetric_hausdorff_2,
        asymmetric_hausdorff_95_1,
        asymmetric_hausdorff_95_2,
        cloud_to_cloud_distance_1,
        cloud_to_cloud_distance_2,
        chamfer_distance
    ]

'''
基于直方图的散度
'''

def kl_divergence_hist_3d(pc_1, pc_2, num_bins=10):
    """
    计算两个三维点云之间的KL散度。
    使用三维直方图来近似概率密度函数。
    """
    # 计算三维直方图的范围
    x_min, y_min, z_min = min(pc_1[:, 0].min(), pc_2[:, 0].min()), min(pc_1[:, 1].min(), pc_2[:, 1].min()), min(pc_1[:, 2].min(), pc_2[:, 2].min())
    x_max, y_max, z_max = max(pc_1[:, 0].max(), pc_2[:, 0].max()), max(pc_1[:, 1].max(), pc_2[:, 1].max()), max(pc_1[:, 2].max(), pc_2[:, 2].max())

    # 计算三维直方图
    hist_1, _ = np.histogramdd(pc_1, bins=num_bins, range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    hist_2, _ = np.histogramdd(pc_2, bins=num_bins, range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    # 归一化直方图
    hist_1 = hist_1 / hist_1.sum()
    hist_2 = hist_2 / hist_2.sum()

    # 避免零值导致的数值问题
    epsilon = 1e-10
    hist_1 = np.maximum(hist_1, epsilon)
    hist_2 = np.maximum(hist_2, epsilon)

    # 计算KL散度
    kl_divergence = stats.entropy(hist_1.flatten(), hist_2.flatten())
    return kl_divergence


def js_divergence_hist_3d(pc_1, pc_2, num_bins=10):
    """
    计算两个三维点云之间的JS散度。
    使用三维直方图来近似概率密度函数。
    """
    # 计算三维直方图的范围
    x_min, y_min, z_min = min(pc_1[:, 0].min(), pc_2[:, 0].min()), min(pc_1[:, 1].min(), pc_2[:, 1].min()), min(pc_1[:, 2].min(), pc_2[:, 2].min())
    x_max, y_max, z_max = max(pc_1[:, 0].max(), pc_2[:, 0].max()), max(pc_1[:, 1].max(), pc_2[:, 1].max()), max(pc_1[:, 2].max(), pc_2[:, 2].max())

    # 计算三维直方图
    hist_1, _ = np.histogramdd(pc_1, bins=num_bins, range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    hist_2, _ = np.histogramdd(pc_2, bins=num_bins, range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    # 归一化直方图
    hist_1 = hist_1 / hist_1.sum()
    hist_2 = hist_2 / hist_2.sum()

    # 避免零值导致的数值问题
    epsilon = 1e-10
    hist_1 = np.maximum(hist_1, epsilon)
    hist_2 = np.maximum(hist_2, epsilon)

    # 计算中间分布 M
    M = 0.5 * (hist_1 + hist_2)

    # 计算KL散度
    kl_divergence_1 = stats.entropy(hist_1.flatten(), M.flatten())
    kl_divergence_2 = stats.entropy(hist_2.flatten(), M.flatten())

    # 计算JS散度
    js_divergence = 0.5 * (kl_divergence_1 + kl_divergence_2)
    return js_divergence


'''
经典分布的参数估计和检验
'''
def estimate_and_test_normal(pc):
    """
    三维正态分布参数估计
    """
    mu_hat = np.mean(pc, axis=0)  # 均值向量
    cov_hat = np.cov(pc, rowvar=False)  # 协方差矩阵
    params = (tuple(mu_hat), tuple(cov_hat.flatten()))  # 将 NumPy 数组转换为元组
    return params, None, None


def estimate_and_test_uniform(pc):
    """
    三维均匀分布参数估计
    """
    a_hat = np.min(pc, axis=0)  # [x_min, y_min, z_min]
    b_hat = np.max(pc, axis=0)  # [x_max, y_max, z_max]
    params = ((a_hat[0], b_hat[0]), (a_hat[1], b_hat[1]), (a_hat[2], b_hat[2]))
    return params, None, None


def estimate_and_test_exponential(pc):
    """
    三维指数分布参数估计
    假设三个维度独立
    """
    lambda_hat_x = 1 / np.mean(pc[:, 0])
    lambda_hat_y = 1 / np.mean(pc[:, 1])
    lambda_hat_z = 1 / np.mean(pc[:, 2])
    params = (lambda_hat_x, lambda_hat_y, lambda_hat_z)
    return params, None, None


def estimate_and_test_laplace(pc):
    """
    三维拉普拉斯分布参数估计
    假设三个维度独立
    """
    mu_hat_x = np.median(pc[:, 0])
    mu_hat_y = np.median(pc[:, 1])
    mu_hat_z = np.median(pc[:, 2])
    b_hat_x = np.mean(np.abs(pc[:, 0] - mu_hat_x))
    b_hat_y = np.mean(np.abs(pc[:, 1] - mu_hat_y))
    b_hat_z = np.mean(np.abs(pc[:, 2] - mu_hat_z))
    params = ((mu_hat_x, b_hat_x), (mu_hat_y, b_hat_y), (mu_hat_z, b_hat_z))
    return params, None, None


'''
经典分布的散度
'''
def kl_divergence_normal(params_1, params_2):
    """
    计算两个三维正态分布之间的KL散度。
    """
    mu1, cov1_flat = params_1
    mu2, cov2_flat = params_2

    # 将协方差矩阵的元组形式转换回三维 NumPy 数组
    cov1 = np.array(cov1_flat).reshape(3, 3)
    cov2 = np.array(cov2_flat).reshape(3, 3)

    inv_cov2 = np.linalg.inv(cov2)
    term1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    term2 = np.trace(inv_cov2 @ cov1)
    term3 = (np.array(mu2) - np.array(mu1)) @ inv_cov2 @ (np.array(mu2) - np.array(mu1))
    return 0.5 * (term1 + term2 + term3 - 3)


def kl_divergence_uniform(params_1, params_2):
    """
    计算两个三维均匀分布之间的KL散度。
    """
    (a1_x, b1_x), (a1_y, b1_y), (a1_z, b1_z) = params_1
    (a2_x, b2_x), (a2_y, b2_y), (a2_z, b2_z) = params_2

    volume1 = (b1_x - a1_x) * (b1_y - a1_y) * (b1_z - a1_z)
    volume2 = (b2_x - a2_x) * (b2_y - a2_y) * (b2_z - a2_z)

    # 添加一个小的偏移量
    epsilon = 1e-10
    volume1 = max(volume1, epsilon)
    volume2 = max(volume2, epsilon)

    return np.log(volume2 / volume1) + volume1 / volume2 - 1


def kl_divergence_exponential(params_1, params_2):
    """
    计算两个三维指数分布之间的KL散度。
    """
    lambda1_x, lambda1_y, lambda1_z = params_1
    lambda2_x, lambda2_y, lambda2_z = params_2

    kl_x = np.log(lambda2_x / lambda1_x) + lambda1_x / lambda2_x - 1
    kl_y = np.log(lambda2_y / lambda1_y) + lambda1_y / lambda2_y - 1
    kl_z = np.log(lambda2_z / lambda1_z) + lambda1_z / lambda2_z - 1
    return kl_x + kl_y + kl_z


def kl_divergence_laplace(params_1, params_2):
    """
    计算两个三维拉普拉斯分布之间的KL散度。
    """
    (mu1_x, b1_x), (mu1_y, b1_y), (mu1_z, b1_z) = params_1
    (mu2_x, b2_x), (mu2_y, b2_y), (mu2_z, b2_z) = params_2

    kl_x = np.log(b2_x / b1_x) + np.abs(mu1_x - mu2_x) / b2_x + b1_x / b2_x - 1
    kl_y = np.log(b2_y / b1_y) + np.abs(mu1_y - mu2_y) / b2_y + b1_y / b2_y - 1
    kl_z = np.log(b2_z / b1_z) + np.abs(mu1_z - mu2_z) / b2_z + b1_z / b2_z - 1
    return kl_x + kl_y + kl_z


def js_divergence_normal(params_1, params_2):
    """
    计算两个三维正态分布之间的JS散度。
    """
    # 计算中间分布的均值和协方差矩阵
    mu1, cov1_flat = params_1
    mu2, cov2_flat = params_2

    cov1 = np.array(cov1_flat).reshape(3, 3)
    cov2 = np.array(cov2_flat).reshape(3, 3)

    mu_m = 0.5 * (np.array(mu1) + np.array(mu2))
    cov_m = 0.5 * (cov1 + cov2)

    # 计算KL散度
    kl_1 = kl_divergence_normal((mu1, cov1_flat), (mu_m, cov_m.flatten()))
    kl_2 = kl_divergence_normal((mu2, cov2_flat), (mu_m, cov_m.flatten()))

    # 计算JS散度
    return 0.5 * (kl_1 + kl_2)


def js_divergence_uniform(params_1, params_2):
    """
    计算两个三维均匀分布之间的JS散度。
    """
    kl_1 = kl_divergence_uniform(params_1, params_2)
    kl_2 = kl_divergence_uniform(params_2, params_1)
    return 0.5 * (kl_1 + kl_2)


def js_divergence_exponential(params_1, params_2):
    """
    计算两个三维指数分布之间的JS散度。
    """
    kl_1 = kl_divergence_exponential(params_1, params_2)
    kl_2 = kl_divergence_exponential(params_2, params_1)
    return 0.5 * (kl_1 + kl_2)


def js_divergence_laplace(params_1, params_2):
    """
    计算两个三维拉普拉斯分布之间的JS散度。
    """
    kl_1 = kl_divergence_laplace(params_1, params_2)
    kl_2 = kl_divergence_laplace(params_2, params_1)
    return 0.5 * (kl_1 + kl_2)


'''
峰度和偏度
'''
def compute_skewness(pc):
    skewness_x = stats.skew(pc[:, 0])
    skewness_y = stats.skew(pc[:, 1])
    skewness_z = stats.skew(pc[:, 2])
    return skewness_x, skewness_y, skewness_z


def compute_kurtosis(pc):
    kurtosis_x = stats.kurtosis(pc[:, 0])
    kurtosis_y = stats.kurtosis(pc[:, 1])
    kurtosis_z = stats.kurtosis(pc[:, 2])
    return kurtosis_x, kurtosis_y, kurtosis_z


'''
参数曲面的拟合和检验
'''
def estimate_and_test_plane(pc):
    """
    对平面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (a, b, c)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    x = pc[:, 0].reshape(-1, 1)
    y = pc[:, 1].reshape(-1, 1)
    z = pc[:, 2]

    # 平面方程：z = ax + by + c
    xy = np.hstack((x, y, np.ones_like(x)))
    params, _, _, _ = np.linalg.lstsq(xy, z, rcond=None)
    a, b, c = params

    # 计算拟合优度
    z_pred = a * x + b * y + c
    r2 = r2_score(z, z_pred)
    rmse = np.sqrt(mean_squared_error(z, z_pred))

    return (a, b, c), r2, rmse


def estimate_and_test_sphere(pc):
    """
    对球面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (x_center, y_center, z_center, radius)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    def residuals(params, x, y, z):
        x_center, y_center, z_center, radius = params
        return np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2) - radius

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    initial_guess = [np.mean(x), np.mean(y), np.mean(z), np.max(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2 + (z - np.mean(z))**2))]

    result = least_squares(residuals, initial_guess, args=(x, y, z))
    params = result.x
    x_center, y_center, z_center, radius = params

    # 计算拟合优度
    residuals = result.fun
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((z - np.mean(z))**2))

    return params, r2, rmse


def estimate_and_test_ellipsoid(pc):
    """
    对椭球面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (x_center, y_center, z_center, a, b, c)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    def residuals(params, x, y, z):
        x_center, y_center, z_center, a, b, c = params
        return (x - x_center)**2 / a**2 + (y - y_center)**2 / b**2 + (z - z_center)**2 / c**2 - 1

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # 计算初始猜测值
    x_center = np.mean(x)
    y_center = np.mean(y)
    z_center = np.mean(z)
    a = np.max(np.abs(x - x_center))
    b = np.max(np.abs(y - y_center))
    c = np.max(np.abs(z - z_center))

    initial_guess = [x_center, y_center, z_center, a, b, c]

    result = least_squares(residuals, initial_guess, args=(x, y, z))
    params = result.x
    x_center, y_center, z_center, a, b, c = params

    # 计算拟合优度
    residuals = result.fun
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((z - np.mean(z))**2))

    return params, r2, rmse


def estimate_and_test_torus(pc):
    """
    对环面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (x_center, y_center, z_center, R, r)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    def residuals(params, x, y, z):
        x_center, y_center, z_center, R, r = params
        term1 = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        term2 = np.sqrt((term1 - R)**2 + (z - z_center)**2)
        return term2 - r

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # 计算初始猜测值
    x_center = np.mean(x)
    y_center = np.mean(y)
    z_center = np.mean(z)
    term1 = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    term2 = np.sqrt((term1 - np.mean(term1))**2 + (z - z_center)**2)
    R = np.mean(term1)
    r = np.mean(term2)

    initial_guess = [x_center, y_center, z_center, R, r]

    result = least_squares(residuals, initial_guess, args=(x, y, z))
    params = result.x
    x_center, y_center, z_center, R, r = params

    # 计算拟合优度
    residuals = result.fun
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((z - np.mean(z))**2))

    return params, r2, rmse


def estimate_and_test_cylinder(pc):
    """
    对圆柱面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (x_center, y_center, radius)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    def residuals(params, x, y):
        x_center, y_center, radius = params
        return np.sqrt((x - x_center)**2 + (y - y_center)**2) - radius

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # 计算初始猜测值
    x_center = np.mean(x)
    y_center = np.mean(y)
    radius = np.max(np.sqrt((x - x_center)**2 + (y - y_center)**2))

    initial_guess = [x_center, y_center, radius]

    result = least_squares(residuals, initial_guess, args=(x, y))
    params = result.x
    x_center, y_center, radius = params

    # 计算拟合优度
    residuals = result.fun
    rmse = np.sqrt(np.mean(residuals**2))

    # 计算 R² 时，使用 y 的方差作为参考
    r2 = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

    return params, r2, rmse


def estimate_and_test_cone(pc):
    """
    对圆锥面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (x_center, y_center, z_center, radius)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    def residuals(params, x, y, z):
        x_center, y_center, z_center, radius = params
        return np.sqrt((x - x_center)**2 + (y - y_center)**2) - radius * np.abs(z - z_center)

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # 计算初始猜测值
    x_center = np.mean(x)
    y_center = np.mean(y)
    z_center = np.min(z)  # 初始猜测为点云的均值
    radius = np.max(np.sqrt((x - x_center)**2 + (y - y_center)**2)) / (np.max(z) - np.min(z))

    initial_guess = [x_center, y_center, z_center, radius]
    # print('initial_guess:', initial_guess)

    result = least_squares(residuals, initial_guess, args=(x, y, z))
    params = result.x
    x_center, y_center, z_center, radius = params
    # print('params:', params)

    # 计算拟合优度
    residuals = result.fun
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((z - np.mean(z))**2))

    return params, r2, rmse


def estimate_and_test_paraboloid(pc):
    """
    对抛物面进行参数回归和拟合优度检验。
    参数:
        pc: n*3 的点云数组，第一列为 x，第二列为 y，第三列为 z。
    返回:
        params: 回归得到的参数 (x_center, y_center, z_center, a)。
        r2: 拟合优度 R²。
        rmse: 均方根误差 RMSE。
    """
    def residuals(params, x, y, z):
        x_center, y_center, z_center, a = params
        return z - (a * (x - x_center)**2 + a * (y - y_center)**2 + z_center)

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    initial_guess = [np.mean(x), np.mean(y), np.mean(z), 1.0]

    result = least_squares(residuals, initial_guess, args=(x, y, z))
    params = result.x
    x_center, y_center, z_center, a = params

    # 计算拟合优度
    residuals = result.fun
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((z - np.mean(z))**2))

    return params, r2, rmse


def compute_VFH(pcd, k=30, temp_dir='temp'):
    """
    计算点云的 VFH 特征。

    参数:
    pcd (o3d.geometry.PointCloud): 输入点云。
    k (int): 计算法向量时的邻域点数量。
    temp_dir (str): 临时文件存储目录。

    返回:
    np.ndarray: 计算得到的 VFH 特征，形状为 (308,)。
    """
    # 确保临时目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 将输入点云保存为临时 PCD 文件
    temp_pcd_path = f"{temp_dir}/temp_pcd.pcd"
    o3d.io.write_point_cloud(temp_pcd_path, pcd)

    # 使用 pclpy 加载点云
    pcl_cloud = pclpy.pcl.PointCloud.PointXYZ()
    pclpy.pcl.io.loadPCDFile(temp_pcd_path, pcl_cloud)

    # 计算法向量
    normals = pclpy.pcl.PointCloud.Normal()
    normal_estimation = pclpy.pcl.features.NormalEstimation.PointXYZ_Normal()
    normal_estimation.setInputCloud(pcl_cloud)
    tree = pclpy.pcl.search.KdTree.PointXYZ()
    normal_estimation.setSearchMethod(tree)
    normal_estimation.setKSearch(k)
    normal_estimation.compute(normals)

    # 计算 VFH 特征
    vfh = pclpy.pcl.features.VFHEstimation.PointXYZ_Normal_VFHSignature308()
    vfh.setFillSizeComponent(True)
    vfh.setInputCloud(pcl_cloud)
    vfh.setInputNormals(normals)
    vfh.setSearchMethod(tree)
    vfhs = pclpy.pcl.PointCloud.VFHSignature308()
    vfh.compute(vfhs)

    # 获取 VFH 特征的直方图
    vfh_histogram = vfhs.histogram[0]
    # print('vfh_histogram:', np.linalg.norm(vfh_histogram))

    return np.array(vfh_histogram)


def VFH_distance(vfh_1, vfh_2):
    """
    计算两个 VFH 特征直方图之间的欧氏距离。

    参数:
    vfh_1 (np.ndarray): 第一个点云的 VFH 特征直方图，形状为 (308,)。
    vfh_2 (np.ndarray): 第二个点云的 VFH 特征直方图，形状为 (308,)。

    返回:
    float: 两个 VFH 特征直方图之间的欧氏距离。
    """
    # 确保输入的 VFH 特征直方图形状正确
    assert vfh_1.shape == (308,) and vfh_2.shape == (308,), "VFH 特征直方图的形状必须为 (308,)"

    # 计算两个 VFH 特征直方图之间的欧氏距离
    distance = np.linalg.norm(vfh_1 - vfh_2)

    return distance


def get_smooth_mesh(pcd, radius=7.5, k=30, radii=None):
    """
    根据点云生成平滑网格，保留所有点（包括未参与三角化的孤立点）
    
    参数:
        pcd: Open3D点云对象
        radius: 球半径参数(默认1.0)
        radii: 球半径列表，用于ball pivoting算法
        
    返回:
        mesh_reconstructed: 原始重建网格
        smoothed_mesh_with_isolated: 平滑处理后的网格（包含孤立点）
    """
        
    # 估计法向量
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k)
    )
    pcd.orient_normals_consistent_tangent_plane(k=k)
        
    # 如果未提供radii，则使用默认值
    if radii is None:
        d = round((27000 / len(pcd.points)) ** 0.5, 1)
        radii = [2 * d, 3 * d, 4 * d, 5 * d, 6 * d]
    
    # 使用ball pivoting方法重建网格
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    # 保留原始重建网格
    mesh_reconstructed = o3d.geometry.TriangleMesh()
    mesh_reconstructed.vertices = mesh.vertices
    mesh_reconstructed.triangles = mesh.triangles
    
    # 提取未参与三角化的点坐标
    triangle_vertices = np.unique(np.asarray(mesh.triangles).flatten())
    all_vertices = np.arange(len(pcd.points))
    non_triangle_vertices = np.setdiff1d(all_vertices, triangle_vertices)
    unreferenced_points = np.asarray(pcd.points)[non_triangle_vertices]
    
    # 移除未引用的顶点
    mesh.remove_unreferenced_vertices()
    
    # 检查网格是否为空（没有三角形）
    if len(mesh.triangles) == 0:
        # 如果网格为空，创建一个简单的空网格并添加所有原始点
        smoothed_mesh_with_isolated = o3d.geometry.TriangleMesh()
        smoothed_mesh_with_isolated.vertices = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        return None, None
    
    # 平滑处理三角化部分
    smoothed_mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    
    # 添加未参与三角化的点
    if len(unreferenced_points) > 0:
        new_vertices = np.vstack((np.asarray(smoothed_mesh.vertices), unreferenced_points))
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        new_mesh.triangles = smoothed_mesh.triangles
        smoothed_mesh_with_isolated = new_mesh
    else:
        smoothed_mesh_with_isolated = smoothed_mesh
    
    return mesh_reconstructed, smoothed_mesh_with_isolated


if __name__ == '__main__':

    pass

    # import numpy as np

    # # 示例点云生成和回归测试
    # np.random.seed(42)  # 设置随机种子以保证结果可重复

    # # 1. 线性曲线测试
    # print("线性曲线测试：")
    # x = np.linspace(-10, 10, 100)
    # y = 0.3 * x - 2 + np.random.normal(0, 0.5, size=x.shape)  # 添加噪声
    # pc = np.column_stack((x, y))
    # params, r2, rmse = estimate_and_test_line(pc)
    # print(f"参数: {params}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # # 2. 圆曲线测试
    # print("\n圆曲线测试：")
    # theta = np.linspace(0, 2 * np.pi, 100)
    # x = 11 + 50 * np.cos(theta)
    # y = -7 + 50 * np.sin(theta) + np.random.normal(0, 1, size=theta.shape)  # 添加噪声
    # pc = np.column_stack((x, y))
    # params, r2, rmse = estimate_and_test_circle(pc)
    # print(f"参数: {params}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # # 3. 椭圆曲线测试
    # print("\n椭圆曲线测试：")
    # theta = np.linspace(0, 2 * np.pi, 100)
    # x = -6 + 50 * np.cos(theta)
    # y = 4 + 20 * np.sin(theta) + np.random.normal(0, 1, size=theta.shape)  # 添加噪声
    # pc = np.column_stack((x, y))
    # params, r2, rmse = estimate_and_test_ellipse(pc)
    # print(f"参数: {params}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # # 4. 二次多项式曲线测试
    # print("\n二次多项式曲线测试：")
    # x = np.linspace(-10, 10, 100)
    # y = 0.2 * x**2 + 2 * x + 3 + np.random.normal(0, 1, size=x.shape)  # 添加噪声
    # pc = np.column_stack((x, y))
    # params, r2, rmse = estimate_and_test_quadratic(pc)
    # print(f"参数: {params}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # # 5. 三次多项式曲线测试
    # print("\n三次多项式曲线测试：")
    # x = np.linspace(-10, 10, 100)
    # y = 0.1 * x**3 + 2 * x**2 + 3 * x + 4 + np.random.normal(0, 1, size=x.shape)  # 添加噪声
    # pc = np.column_stack((x, y))
    # params, r2, rmse = estimate_and_test_cubic(pc)
    # print(f"参数: {params}, R²: {r2:.4f}, RMSE: {rmse:.4f}")