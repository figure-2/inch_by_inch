풀이 1
def solution(my_string, letter):
    answer = ''

    for string in my_string:
        if string != letter:
            answer += string
    return answer

# 모음 제거 하기 문제랑 같은 문제


풀이 2
def solution(my_string, letter):
    answer = ''

    answer = my_string.replace(letter, '')
    return answer