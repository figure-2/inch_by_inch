# 매개변수에서 중앙값 리턴
# sort를 통해서 오름 차순 정렬을 하고  중앙 

def solution(array):
    array.sort()
    answer = array[len(array)//2]
    return answer


# 1. 먼저 array를 오름차순으로 정렬을 시킨다 

# 2. array가 중앙값이 오려면 array의 길이가 3일때 array[1]가 나와야하고 길이가 5일때 array[3]이 되어야한다.

# 그래서 array의 길이가 홀수라 주어질때 2로 나누면 이렇게 된다.

# array = 3 , 5  ,7 일때 몫의 값은 1, 2 ,3 ... 이런식으로 올라간다. 


def solution(array):
    array.sort()
    n = len(array)

    if n % 2 == 1:
     i = array[n // 2]
    else:
       i1 = array[(n // 2) -1]
       i2 = array[n // 2]
       i = (i1 + i2) / 2 
    
    return i


# def solution(array):
#     array.sort()
#     n = len(array)

#     if n % 2 == 1:
#      i = array[n // 2]
#     else:
#        i1 = array[(n // 2) -1]
#        i2 = array[n // 2]
#        i = (i1 + i2) / 2

#     return i