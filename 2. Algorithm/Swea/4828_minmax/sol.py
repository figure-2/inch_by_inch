import sys
sys.stdin = open('input.txt')

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    numbers = list(map(int, input().split()))
    # #split로 조깨고 map으로 숫자로 만들어주고 list로 바꾼다

    # #print(N, numbers)'

    # #정렬 후 큰수 작은수 뽑아서 연산

    # numbers.sort()
    # print(numbers)
    # result = numbers[-1] - numbers[0]
    # print(f'#{tc} {result}')

    # min_number = 100000000
    # max_number = 0
    # 위와 아래 차이가 없다
    min_number = numbers[0]
    max_number = numbers[0]

    for number in numbers:
        if min_number > number:
            min_number = number

        if max_number < number:
            max_number = number
     result = numbers[-1] - numbers[0]
     print(f'#{tc} {result}')

