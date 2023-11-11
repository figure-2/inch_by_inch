# 정수 배열 arr


def solution(arr):
    stk = []
    i = 0
    
    while i < len(arr): # i가 arr의 길이보다 작으면 다음 작업을 반복
        if len(stk) == 0:
            stk.append(arr[i])
            i += 1
        elif len(stk) > 0 and stk[-1] < arr[i]:
            stk.append(arr[i])
            i += 1
        elif len(stk) > 0 and stk[-1] >= arr[i]:
            stk.pop()
    
    return stk