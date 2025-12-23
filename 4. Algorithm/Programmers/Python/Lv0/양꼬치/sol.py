def solution(n, k):
    kushi = 12000
    drink = 2000
    answer = kushi * n + drink * k
    if n >= 10 :
        answer -= drink * int(n/10)
    return answer