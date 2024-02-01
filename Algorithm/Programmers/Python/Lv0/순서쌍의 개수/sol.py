풀이 1
def solution(n):
    answer = 0
    
    for num in range(1, n+1):
        if n % num == 0:
            answer += 1

    return answer

풀이 2 #계산량을 절반으로 줄이기 위한 코드
def solution(n):
    answer = 0    
    for num in range(1, int(n**0.5) + 1):  #n**0.5 루트가 씌어진 상태,그리고 루트에서는 소수점이 나올수 있기 때문에 in로 변환해준다
        if n % num == 0:
            answer += 2

            if num * num == n:
                answer -= 1

    return answer