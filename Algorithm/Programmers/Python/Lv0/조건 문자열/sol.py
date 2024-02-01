# n 과 m 문자열 
# ineq 와 eq가 주어짐
# ineq 는  < , >
# eq 는  = , !


def solution(ineq, eq, n, m):
    if eq == "!" :
        if ineq == "<" :
             return int(n<m)
        else:
            return int(n>m)
    else:
        if ineq == "<" :
             return int(n<=m)
        else:
            return int(n>=m)
