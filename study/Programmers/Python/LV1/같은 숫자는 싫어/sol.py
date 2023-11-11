# def solution(arr):
#     answer = []
#     # [실행] 버튼을 누르면 출력 값을 볼 수 있습니다.
#     print('Hello Python')
#     return answer


# set 리스트에서 중복된 값을 삭제 할 수 있다.


def solution(arr):
    answer = [arr[0]]
    for i in range([1], len(arr)):
        if arr[i] != arr[i-1]:
            answer.append(arr[i])
    return answer


