# def solution(dartResult):
#     answer = 0
#     return answer



# {Single(S), Double(D), Triple(T) : ( '점수1' , '점수'2 , '점수3' ) }
# 스타상(*)  해당 점수와 이던번 던진 점수의 2배
# 아차상(#) 해당 점수는 -
    
def solution(dartResult):
    dart = []
    dartResult = dartResult.replace("10", "A")
    bonus = {'S': 1, 'D': 2, 'T': 3}
    
    for i in dartResult:
        if i.isdigit() or i=='A':
            dart.append(10 if i == 'A' else int(i))
        elif i in ('S', 'D', 'T'):
            num = dart.pop()
            dart.append(num ** bonus[i])
        elif i == '#':
            dart[-1] *= -1
        elif i == '*':
            num = dart.pop()
            if len(dart):
                dart[-1] *= 2
            dart.append(2 * num)
    return sum(dart)


# 나는 이 문제를 스택 문제로 보았다.

# 주어지는 문자열 dartResult는 숫자, (S,D,T) 중 하나, (#, *) 중 하나로 구성되며 #, *은 없을 수도 있고, 둘 다 있을 수도 있다.
# 우선 점수에 해당하는 숫자는 0 ~ 10인데 dartResult를 순회하며 확인하려 했더니 10은 두자리여서 예외 처리를 해야했다. 그래서 10을 다른 문자인 'A'로 바꿔버렸다.

# dartResult = dartResult.replace("10", "A")

# 그리고 S, D, T는 점수 계산식인데 각각 1제곱, 2제곱, 3제곱이다. 그래서 이 값을 딕셔너리로 관리하기 위해 bonus 딕셔너리를 선언했다.
# bonus = {'S': 1, 'D': 2, 'T': 3}

# 이렇게 전처리를 마친 후 dartResult를 순회하며 하나씩 처리한다.
# i가 숫자라면 그대로 스택에 넣는다. (A는 10으로 바꿔서 스택에 넣는다)
# 이때 정수로 변환해서 스택에 넣어준다.

# i가 S, D, T중 하나라면 스택에서 마지막 들어간 아이템을 하나 꺼내서 bonus[i] 만큼 제곱해서 다시 스택에 넣는다. 
# i가 #이라면 스택의 마지막 요소에 -1를 곱한다.
# i가 *이라면
# 선 스택에서 마지막 요소를 하나 꺼낸다.
# 만약 요소를 꺼낸 뒤 스택에 요소가 더 남아있다면 그 마지막 요소에 2를 곱하고,
# 꺼냈던 요소에도 2를 곱해서 다시 넣어준다.

#  위 과정이 모두 끝나면 스택에는 숫자만 남아있을 것이다
# 그 숫자들을 모두 더해주면 점수가 된다.