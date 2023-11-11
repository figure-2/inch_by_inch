def solution(my_string, overwrite_string, s):
    answer = ''
    if len(my_string[s:]) > len(overwrite_string):
        answer = my_string[0:s] + overwrite_string + my_string[s+len(overwrite_string):]
    else:
        answer = my_string[0:s] + overwrite_string
    
    return answer


    
def solution(my_string, overwrite_string, s):
    L1 = list(my_string)
    L2 = list(overwrite_string)
    j = 0
    for i in range(s, s+len(L2)):
        L1[i] = L2[j]
        j += 1
    answer = ''.join(L1)
    return answer
    