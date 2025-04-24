import open3d as o3d
import numpy as np


def generate_pc_from_mesh(mesh, sample_method="poisson", pc_num=10000, sigma=0.5):

    if sample_method == 'uniform':
        pcd = mesh.sample_points_uniformly(number_of_points=pc_num)
    else:
        pcd = mesh.sample_points_poisson_disk(number_of_points=pc_num)

    points = np.asarray(pcd.points)

    if sigma is None:
        return points
    
    else:
        noise = np.random.normal(0, sigma, size=points.shape)
        points += noise
        return points


def create_sphere(radius, resolution=20):
    '''
    生成一个指定半径和分辨率的等腰直角三角形棱柱mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    return mesh


def create_cylinder(radius, height, resolution=20):
    '''
    生成一个指定半径和分辨率的圆柱mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    return mesh


def create_equilateral_triangular_prism(side_length, prism_height):
    '''
    生成一个指定边长和高的等边三角形棱柱mesh
    '''
    height = side_length * np.sqrt(3) / 2
    vertices = np.array([
        [0, 0, 0],
        [side_length, 0, 0],
        [side_length / 2, height, 0],
        [0, 0, prism_height],
        [side_length, 0, prism_height],
        [side_length / 2, height, prism_height]
    ])
    triangles = np.array([
        [0, 1, 2],
        [3, 5, 4],
        [0, 3, 4],
        [0, 4, 1],
        [1, 4, 5],
        [1, 5, 2],
        [2, 5, 3],
        [2, 3, 0]
    ])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def create_isosceles_right_triangular_prism(side_length, prism_height):
    '''
    生成一个指定高度和边长的等腰直角三角形棱柱mesh
    '''

    # 生成各个顶点
    vertices = o3d.utility.Vector3dVector(np.array([
        [0, 0, 0],
        [side_length, 0, 0],
        [0, side_length, 0],
        [0, 0, prism_height],
        [side_length, 0, prism_height],
        [0, side_length, prism_height]
    ]))

    # 生成各个面（连接）
    triangles = o3d.utility.Vector3iVector(np.array([
        [0, 1, 2], # 下底面
        [3, 4, 5], # 上底面
        [0, 3, 4], # 侧面1
        [0, 4, 1], # 侧面1
        [1, 4, 5], # 侧面2
        [1, 5, 2], # 侧面2
        [2, 5, 3], # 侧面3
        [2, 3, 0], # 侧面3
    ]))

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = vertices
    mesh.triangles = triangles

    return mesh


def create_cube(side_length):
    '''
    生成一个指定边长的正方体mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_box(width=side_length, height=side_length, depth=side_length)
    return mesh


def create_quadrangular_prism(width_a, width_b, height):
    '''
    生成一个指定长宽高和分辨率的四棱柱（长方体）mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_box(width=width_a, height=width_b, depth=height)
    return mesh


def create_pentagonal_prism(side_length, prism_height):
    '''
    生成一个指定边长和高的五棱柱mesh
    '''
    angle = 2 * np.pi / 5
    vertices_base = []
    for i in range(5):
        x = side_length * np.cos(i * angle)
        y = side_length * np.sin(i * angle)
        vertices_base.append([x, y, 0])

    vertices_top = [[x, y, prism_height] for x, y, _ in vertices_base]

    vertices = vertices_base + vertices_top

    # Define triangles
    triangles = np.array([
        [0, 5, 6],  [0, 6, 1],
        [1, 6, 7],  [1, 7, 2],
        [2, 7, 8],  [2, 8, 3],
        [3, 8, 9],  [3, 9, 4],
        [4, 9, 5],  [4, 5, 0],
        [0, 1, 2], [0, 2, 3], [0, 3, 4],
        [5, 6, 7], [5, 7, 8], [5, 8, 9],

    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def create_hexagonal_prism(side_length, prism_height):
    '''
    生成一个指定边长和高的六棱柱mesh
    '''
    angle = 2 * np.pi / 6
    vertices_base = []
    for i in range(6):
        x = side_length * np.cos(i * angle)
        y = side_length * np.sin(i * angle)
        vertices_base.append([x, y, 0])

    vertices_top = [[x, y, prism_height] for x, y, _ in vertices_base]

    vertices = vertices_base + vertices_top

    # Define triangles
    triangles = np.array([
        [0, 6, 7],  [0, 7, 1],
        [1, 7, 8],  [1, 8, 2],
        [2, 8, 9],  [2, 9, 3],
        [3, 9, 10],  [3, 10, 4],
        [4, 10, 11],  [4, 11, 5],
        [5, 11, 6],  [5, 6, 0],
        [0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5],
        [6, 7, 8], [6, 8, 9], [6, 9, 11], [9, 10, 11],
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def create_l_beam(width, height, thickness):
    '''
    生成一个指定L边长、高度和厚度的L梁mesh
    '''
    bottom_vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, thickness, 0],
        [thickness, thickness, 0],
        [thickness, width, 0],
        [0, width, 0],

    ])
    top_vertices = bottom_vertices + np.array([0, 0, height])
    vertices = np.concatenate((bottom_vertices, top_vertices))

    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5],
        [6, 7, 8], [6, 8, 9], [6, 9, 10], [6, 10, 11],
        [0, 6, 7], [0, 7, 1], [1, 7, 8], [1, 8, 2], 
        [2, 8, 9], [2, 9, 3], [3, 9, 10], [3, 10, 4], 
        [4, 10, 11], [4, 11, 5], [5, 11, 6], [5, 6, 0],
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def create_t_beam(width, height, thickness):
    '''
    生成一个指定T的边长、高度和厚度的T形梁mesh
    '''
    bottom_vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, thickness, 0],
        [(width + thickness)/2, thickness, 0],
        [(width + thickness)/2, width, 0],
        [(width - thickness)/2, width, 0],
        [(width - thickness)/2, thickness, 0],
        [0, thickness, 0],

    ])
    top_vertices = bottom_vertices + np.array([0, 0, height])
    vertices = np.concatenate((bottom_vertices, top_vertices))

    triangles = np.array([
        [0, 1, 3], [1, 2, 3], [0, 3, 6], [0, 6, 7], [3, 4, 5], [3, 5, 6],
        [8, 9, 11], [9, 10, 11], [8, 11, 14], [8, 14, 15], [11, 12, 13], [11, 13, 14],
        [0, 8, 9], [0, 9, 1], [1, 9, 10], [1, 10, 2],
        [2, 10, 11], [2, 11, 3], [3, 11, 12], [3, 12, 4], 
        [4, 12, 13], [4, 13, 5], [5, 13, 14], [5, 14, 6], 
        [6, 14, 15], [6, 15, 7], [7, 15, 8], [7, 8, 0],
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def create_u_beam(width, height, thickness):
    '''
    生成一个指定梭长的八面体mesh
    '''
    bottom_vertices = np.array([
        [0, 0, 0], 
        [width, 0, 0], [width, width, 0],
        [width - thickness, width, 0], [width - thickness, thickness, 0], 
        [thickness, thickness, 0], [thickness, width, 0], 
        [0, width, 0],
    ])

    top_vertices = bottom_vertices + np.array([0, 0, height])
    vertices = np.concatenate((bottom_vertices, top_vertices))

    triangles_bottom = np.array([
        [0, 1, 4], [0, 4, 5], [1, 2, 4], [2, 3, 4], [5, 6, 7], [0, 5, 7]
    ])
    triangles_top = triangles_bottom + 8
    triangles_side = np.array([
        [0, 8, 9], [0, 9, 1], [1, 9, 10], [1, 10, 2],
        [2, 10, 11], [2, 11, 3], [3, 11, 12], [3, 12, 4], 
        [4, 12, 13], [4, 13, 5], [5, 13, 14], [5, 14, 6], 
        [6, 14, 15], [6, 15, 7], [7, 15, 8], [7, 8, 0],
    ])

    triangles = np.concatenate((
        triangles_bottom, 
        triangles_side, 
        triangles_top
    ))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def create_h_beam(width, height, thickness):
    '''
    生成一个指定边长、高度和厚度的H梁mesh
    '''
    bottom_vertices = np.array([
        [0, 0, 0], 
        [width, 0, 0], [width, thickness, 0], 
        [(width + thickness)/2, thickness, 0], [(width + thickness)/2, width - thickness, 0], 
        [width, width - thickness, 0], [width, width, 0], 
        [0, width, 0], [0, width - thickness, 0],
        [(width - thickness)/2, width - thickness, 0], [(width - thickness)/2, thickness, 0], 
        [0, thickness, 0],
    ])

    top_vertices = bottom_vertices + np.array([0, 0, height])
    vertices = np.concatenate((bottom_vertices, top_vertices))

    triangles_bottom = np.array([
        [0, 1, 3], [1, 2, 3], [0, 3, 10], [0, 10, 11], 
        [3, 4, 9], [3, 9, 10],
        [4, 5, 6], [4, 6, 9], [6, 7, 9], [7, 8, 9]
    ])
    triangles_top = triangles_bottom + 12
    triangles_side = np.array([
        [0, 12, 13], [0, 13, 1], [1, 13, 14], [1, 14, 2],
        [2, 14, 15], [2, 15, 3], [3, 15, 16], [3, 16, 4], 
        [4, 16, 17], [4, 17, 5], [5, 17, 18], [5, 18, 6], 
        [6, 18, 19], [6, 19, 7], [7, 19, 20], [7, 20, 8], 
        [8, 20, 21], [8, 21, 9], [9, 21, 22], [9, 22, 10], 
        [10, 22, 23], [10, 23, 11], [11, 23, 12], [11, 12, 0],
    ])

    triangles = np.concatenate((
        triangles_bottom, 
        triangles_side, 
        triangles_top
    ))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def create_cone(radius, height, resolution=20):
    '''
    生成一个指定半径、高和分辨率的圆锥mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=resolution)
    return mesh


def create_torus(torus_radius, tube_radius, radial_resolution=30, tubular_resolution=20):
    '''
    生成一个指定整体半径、管道半径和分辨率的甜甜圈mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=torus_radius, tube_radius=tube_radius, radial_resolution=radial_resolution, tubular_resolution=tubular_resolution)
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    mesh = o3d.geometry.TriangleMesh.rotate(mesh, R)
    return mesh


def create_tetrahedron(radius):
    '''
    生成一个指定梭长的四面体mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius)
    return mesh


def create_octahedron(radius):
    '''
    生成一个指定梭长的八面体mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_octahedron(radius=radius)
    return mesh


def create_mobius(twists=1, raidus=1, flatness=1, width=1, scale=100):
    '''
    生成一个指定半径和分辨率的圆柱mesh
    '''
    mesh = o3d.geometry.TriangleMesh.create_mobius(twists=twists, raidus=raidus, flatness=flatness, width=width, scale=scale)
    return mesh
