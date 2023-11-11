# 매개변수는 소문자로 구성 공백 없이 있고 영어로 써있는 숫자를 아라비아 숫자로 표현해라
#zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"

def solution(numbers):
    numbers2 = {'zero' : 0, 'one':1, 'two':2, 'three' : 3, 'four':4, 'five':5,'six' : 6, 'seven' : 7, 'eight':8,'nine':9}
    # dictionary에 key와 value 를 지정한다.

    for a, b in numbers2.items():  # dictionary 함수 중 key 와 value 를 가져오는 .items() 함수를 사용해서 a 에는 영문 / b 에는 숫자 를 입력
        numbers = numbers.replace(a, str(b)) # 문자열을 대치하는 .replace 함수를 써서 a 를 b 로 대치 후 다시 numbers에 저장

    return int(numbers)


# enumerate() 라는 함수도 있네 들어본적은 있는데 기억이... 이 함수를 쓰면 리스트에서 인덱스 와 객체(?)를 순서대롤 나열???


# for entry in enumerate(['A', 'B', 'C']):
#     print(entry)

# (0, 'A')
# (1, 'B')
# (2, 'C')