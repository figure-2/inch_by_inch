# 2 차원 정부 배열 queries / s, e의 형태



def solution(my_string, queries):

    for s, e in queries:
        my_string = my_string[:s] + my_string[s:e+1][::-1] + my_string[e+1:]
    return my_string



# 이 함수는 문자열(my_string)과 쿼리(queries) 두 가지 입력을 받습니다. 쿼리는 시작 인덱스(s)와 끝 인덱스(e)의 튜플로 구성된 리스트입니다.

# 먼저, 각 쿼리를 반복합니다. `for s, e in queries:`라는 문장은 queries 리스트에 있는 각 튜플을 순회하며, 튜플의 첫 번째 요소를 s에 할당하고 두 번째 요소를 e에 할당합니다. 예를 들어, queries가 [(1, 3), (4, 6)]라면 첫 번째 반복에서 s는 1이고 e는 3이 될 것입니다.

# 다음으로 문자열 슬라이싱과 결합을 사용하여 my_string을 변형합니다.
# `my_string = my_string[:s] + my_string[s:e+1][::-1] + my_string[e+1:]` 이 부분은 다음과 같이 동작합니다:

# - `my_string[:s]`: 문자열의 시작부터 s 인덱스 전까지의 부분
# - `my_string[s:e+1][::-1]`: s 인덱스부터 e 인덱스까지의 부분을 역순으로 정렬
# - `my_string[e+1:]`: e 인덱스 다음부터 문자열의 끝까지

# 세 부분을 합쳐서 새로운 문자열을 만들고 이것을 다시 my_string에 할당합니다.

# 따라서 이 함수는 주어진 범위 내에서 원래 문자열의 부분들을 역순으로 바꾸고 그 결과를 반환하는 작업을 수행합니다.


# replace로 풀기랑 

# s,e 	queries 매소드
# 따라서 이 함수는 주어진 범위 내에서 원래 문자열의 부분들을 역순으로 바꾸고 그 결과를 반환하는 작업을 수행합니다.
