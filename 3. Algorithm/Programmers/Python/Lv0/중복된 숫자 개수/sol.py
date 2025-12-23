# array 에 매개변수 정수 n이 몇개 있는지
# 조건문, 

# def solution(array, n):
#     te = []
    
#     if n in array:
#         te += 1
#     return te

def solution(array, n):
    answer = array.count(n)
    return answer

# count()
# count() : 파이썬 리스트의 특정 요소 개수 구하기
# [리스트 이름]. count(찾아야 할 특정 요소)