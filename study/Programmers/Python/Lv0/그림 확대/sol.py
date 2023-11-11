# 제곱을 써야 하는것 같은데 아닌가 그냥 곱하기 인거 같은디


def solution(picture, k):

    answer = []

    for i in picture:
        l = ''
        for m in i:
            l += m*k
        for count in range(k):
            answer.append(l)

    return answer



def solution(picture, k):
    answer = []
    
    for row in picture: # 이미지의 한 줄을 가져온다.
        resized = ''
        
        for pixel in row:
            resized += pixel * k # 한 픽셀을 k배 만큼 가로로 늘린다.
        
        for _ in range(k):
            answer.append(resized) # 가로로 늘려진 이미지 한 줄을 k배 만큼 세로로 늘린다. 
    
    return answer


def solution(picture, k):
    answer = []
    
    for i in picture:
        char = ""
        for j in i:
            char += j*k
        for count in range(k):
            answer.append(char)
    
    return answer