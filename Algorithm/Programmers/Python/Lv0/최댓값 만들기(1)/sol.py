풀이 1
def solution(numbers):
    answer = 0

    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            result = numbers[i] * numbers[j]
            
            if answer < result:
                answer = result

    return answer


풀이 2
def solution(numbers):
    answer = 0

    numbers.sort()  # 오름 차순으로 정렬
    answer = numbers[-1] * numbers[-2]

    return answer

풀이 2
def solution(numbers):
    answer = 0

    numbers.sort(reverse=True)  # 내림차순으로 정렬
    answer = numbers[0] * numbers[1]

    return answer