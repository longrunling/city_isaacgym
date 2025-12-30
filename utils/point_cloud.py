import numpy as np

def calculate_yaw(point1, point2):

    point1 = np.array(point1).squeeze()
    point2 = np.array(point2).squeeze()
    vector = point2 - point1
    angle = np.arctan2(vector[1], vector[0])
    return angle

def find_nearest_point(point, point_cloud):

    distance = np.linalg.norm(np.array(point) - np.array(point_cloud), axis = 1)
    idx = np.where(distance == min(distance))[0][0]
    return point_cloud[idx]

def find_common_points_exact(cloud1, cloud2):
    # 将cloud2转换为元组集合（加速查找）
    cloud2_set = set(map(tuple, cloud2))
    
    # 检查cloud1中的每个点是否在cloud2_set中
    mask = np.array([tuple(point) in cloud2_set for point in cloud1])
    
    return cloud1[mask]