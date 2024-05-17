import numpy as np

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
cross_product = calculate_ccw_angle(start_point, center_point, end_point)

# 외적의 부호를 통해 두 벡터 사이의 각이 시계 방향인지 판별합니다.
if cross_product > 0:
    print("둔각입니다.")
elif cross_product < 0:
    print("예각입니다.")
else:
    print("직각입니다.")
