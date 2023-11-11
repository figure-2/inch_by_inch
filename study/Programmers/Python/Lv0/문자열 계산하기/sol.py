def solution(my_string):
    my_string = my_string.split()
    answer = int(my_string[0])
    
    for i in range(len(my_string)):
        if my_string[i] == '+':
            answer += int(my_string[i+1])
        elif my_string[i] == '-':
            answer -= int(my_string[i+1])
        else:
            continue
            
    return answer



# 이 코드는 입력된 문자열 my_string을 공백을 기준으로 분리하고, 그 결과를 사용하여 덧셈과 뺄셈을 수행하는 함수입니다. 아래에서 코드를 단계별로 설명하겠습니다.

# my_string을 공백을 기준으로 나누고, 나눈 결과를 리스트로 저장합니다. 이 리스트에는 숫자와 연산자(+ 또는 -)가 번갈아 나오게 됩니다.

# answer 변수에 첫 번째 숫자를 정수로 변환하여 저장합니다. 이 변수는 최종적으로 결과를 저장할 변수입니다.

# 반복문을 사용하여 리스트를 순회합니다. 리스트의 각 요소를 확인하면서 다음을 수행합니다:

# 요소가 '+'인 경우, 현재의 answer에 다음 숫자를 더합니다.
# 요소가 '-'인 경우, 현재의 answer에서 다음 숫자를 뺍니다.
# 그 외의 경우에는 아무 작업도 하지 않고 다음 요소로 넘어갑니다.
# 반복문이 끝나면, answer에는 입력된 수식을 계산한 결과가 저장됩니다.

# 마지막으로 answer를 반환하여 결과를 출력합니다.

# 이 함수는 주어진 수식을 계산하여 결과를 반환합니다. 예를 들어, "5 + 3 - 2"와 같은 문자열을 입력하면, 이 함수는 5 + 3 - 2를 계산하여 6을 반환합니다.