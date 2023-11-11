# # 모든 자연수의 합을 return 그러면 매개변수에서 숫자를 뽑아서 리스트에 넣거 그 리스트에 있는 값을 더하기

# def solution(my_string):


# def solution(my_string):
#     numbers = []  # 자연수들을 저장할 리스트 생성
#     num = ""  # 자연수를 만들기 위한 임시 문자열 변수
#     for char in my_string:
#         if char.isdigit():  # 문자열이 숫자인 경우
#             num += char  # 임시 문자열에 추가
#         else:  # 문자열이 숫자가 아닌 경우
#             if num:  # 이전까지 숫자가 나온 적이 있다면
#                 numbers.append(int(num))  # 자연수로 변환하여 리스트에 추가
#                 num = ""  # 임시 문자열 초기화
#     if num:  # 마지막 문자열이 숫자인 경우
#         numbers.append(int(num))  # 자연수로 변환하여 리스트에 추가
#     return sum(numbers)  # 자연수들의 합 반환

def solution(my_string):
    numbers = []  # 자연수들을 저장할 리스트 생성
    for char in my_string:
        if char.isdigit():  # 문자열이 숫자인 경우
            numbers.append(int(char))  # 숫자를 자연수로 변환하여 리스트에 추가
            # numbers += int(char)
    return sum(numbers)  # 자연수들의 합 반환