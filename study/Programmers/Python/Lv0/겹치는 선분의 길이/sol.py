def solution(lines):
    table = [set([]) for _ in range(200)] # -100~100까지 각 선분들의 등장 count 초기화
    for index, line in enumerate(lines):
        x1, x2 = line
        for x in range(x1, x2):
            table[x + 100].add(index) # 선분에 음수가 들어있을 수도 있으므로 +100

    answer = 0
    for line in table:
        if len(line) > 1:
            answer += 1

    return answer


def solution(lines):
    answer = 0
    count = [0 for _ in range(200)] # -100 ~ 100 까지의 범위에서 해당 점에 선분이 그어진 횟수
    for line in lines:
        for i in range(line[0], line[1]): 
            count[i + 100] += 1
    answer += count.count(2) # 두 개 이상 겹친 점
    answer += count.count(3) # 세 개 이상 겹친 점
    return answer