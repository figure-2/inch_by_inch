# 정수 배열 arr


def solution(arr, k):
    answer = []

    for item in arr:
        if item not in answer:
            answer.append(item)
        if len(answer) == k:
            break

    if len(answer) < k:
        answer += [-1] * (k - len(answer))

    return answer


def solution(arr, k):
    answer = []
    for i in arr: #반복분을 돌려서 
        if i not in answer:
            answer.append(i)

        if len(answer) == k :
            return answer

    if len(answer) != k:
        for j in range(k-len(answer)) :
            answer.append(-1)

    return answer