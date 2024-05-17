import numpy as np

def calculate_angle(start_point, center_point, end_point):
    # 시작점과 중심점을 이용하여 첫 번째 벡터를 계산합니다.
    vector1 = np.array([center_point[0] - start_point[0], center_point[1] - start_point[1]])
    
    # 중심점과 바늘 끝점을 이용하여 두 번째 벡터를 계산합니다.
    vector2 = np.array([end_point[0] - center_point[0], end_point[1] - center_point[1]])
    
    # 두 벡터의 내적을 계산합니다.
    dot_product = np.dot(vector1, vector2)
    
    # 두 벡터의 크기를 계산합니다.
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # 내적을 이용하여 두 벡터 사이의 각도를 계산합니다.
    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    angle_rad = np.arccos(cos_theta)
    
    # 라디안을 각도로 변환합니다.
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_ccw_angle(start_point, center_point, end_point):
    # 시작점과 중심점을 이용하여 첫 번째 벡터를 계산합니다.
    vector1 = np.array([center_point[0] - start_point[0], center_point[1] - start_point[1]])
    
    # 중심점과 끝점을 이용하여 두 번째 벡터를 계산합니다.
    vector2 = np.array([end_point[0] - center_point[0], end_point[1] - center_point[1]])
    
    # 외적을 계산합니다.
    cross_product = np.cross(vector1, vector2)
    
    return cross_product

# 예시 점들
start_point = (573, 491)
center_point = (693, 373)
end_point = (816, 486)

# 외적 계산


# x0, y0 = 693, 373
# x1, y1 = 573, 491
# x2, y2 = 816,486

# 각도 계산
angle = calculate_angle(start_point, center_point, end_point)
cross_product = calculate_ccw_angle(start_point, center_point, end_point)

print("바늘이 가르키는 각도:", angle)
# 외적의 방향을 통해 둔각인지 예각인지 판별합니다.
if cross_product > 0:
    print("둔각입니다.")
elif cross_product < 0:
    print("예각입니다.")
else:
    print("직각입니다.")