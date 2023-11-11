
def solution(spell, dic):
    a = sorted(spell)
    b = []
    for i in dic :
        b.append(sorted(i))
    if a in b:
        answer = 1
    else:
        answer = 2
    return answer