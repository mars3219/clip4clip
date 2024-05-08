import math

def angle_greater_than_180(x0, y0, x1, y1, x2, y2):
    # 시작점과 중심점의 벡터
    vector_start_to_center = (x0 - x1, y0 - y1)
    
    # 시작점과 끝점의 벡터
    vector_start_to_end = (x2 - x1, y2 - y1)
    
    # 외적 계산
    cross_product = vector_start_to_center[0] * vector_start_to_end[1] - vector_start_to_end[0] * vector_start_to_center[1]
    
    # 외적의 부호에 따라 각이 180도를 넘는지 판단//180 넘으면 음수
    return cross_product > 0

def calculate_angle(x0, y0, x1, y1, x2, y2):
    # 시작점과 중심점에서 끝점까지의 벡터를 구합니다.
    vector1 = [x1 - x0, y1 - y0]
    vector2 = [x2 - x0, y2 - y0]

    # 두 벡터의 내적을 구합니다.
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    value = dot_product / (math.sqrt(vector1[0]**2 + vector1[1]**2) * math.sqrt(vector2[0]**2 + vector2[1]**2))
    value = max(-1, min(1, value))

    # 각도를 구합니다.
    angle = math.acos(value)

    obtuse_angle = angle_greater_than_180(x0, y0, x1, y1, x2, y2)

    print(obtuse_angle)

    # 각도를 180도로 변환합니다.

    angle_deg = math.degrees(angle)

    if not obtuse_angle:
        angle_deg = 360 - angle_deg

    return angle_deg

# 예시 시작점, 중심점, 끝점
x0, y0 = 0, 0  # 중심점
x1, y1 = -5*math.sqrt(2), -5*math.sqrt(2)  # 시작점
x2, y2 = 5*math.sqrt(2), 5*math.sqrt(2)
# x2, y2 = 0, 10
# x2, y2 = 10, 0  # 끝점

# 각을 계산합니다.
angle = calculate_angle(x0, y0, x1, y1, x2, y2)

# 180도를 넘는지 확인합니다.
if angle > 180:
    print("180도를 넘습니다.", angle)
else:
    print("180도를 넘지 않습니다.", angle)
