# 가로 세로
# 가로 x1 x2
# 세로 y1 y2

def solution(dots): 
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]
    
    w = max(x) - min(x)
    h = max(y) - min(y)
    area = w*h
    return area