# 0 과 1 arr
# arr을 stk
# i는 0 / 
# stk 빈배열 arr[i] fmf stk에 추가하고 i 에 1들 더한다


def solution(arr):
    stk = []
    
    for i in range(len(arr)):
        if len(stk) == 0:
            stk.append(arr[i])
        else:
            if stk[-1] == arr[i]:
                stk.pop()
                i += 1
            elif stk[-1] != arr[i]:
                stk.append(arr[i])
                i += 1
    
    if len(stk) == 0:
        return [-1]
    
    answer = stk
    
    return answer