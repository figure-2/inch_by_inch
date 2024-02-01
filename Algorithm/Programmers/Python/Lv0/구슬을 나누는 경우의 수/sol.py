# 구슬의 갯수 balls
# 친구들에게 나누어줄 갯수 share
# n 개 중 m개를 뽑는 경우의 수 n! / (n-m)! * m!

# def fact(n):
#     result = 1
#     while n > 1:
#         result *= n # result = result * n
#         n -= 1  # n = n - 1
#     return result

from math import factorial as fact


def fact(num):
    a = 1
    for i in range(1, num+1):
        a *= i
    return a

def solution(balls, share):
    answer = 0
    b = fact(balls)
    s = fact(share)
    bs = fact(balls-share)
    
    answer = b / (bs * s)
    
    return answer