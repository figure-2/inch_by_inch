        
def solution(n):
    str_n = str(n)
    answer = 0

    for i in str_n:
        answer += int(i)
    return answer

풀이 2
def solution(n):
    return sum([int(i) for i in str(n)])
