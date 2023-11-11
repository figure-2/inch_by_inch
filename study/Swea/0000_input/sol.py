# import sys
# sys.stdin = open('input.txt')

# #stdin 의 뜻 standard in /  표준

# # N = int(input()) #  input 되는 값이 숫자지만 문자로 인식하기 때문에 int()로 숫자로 변환해줌

# # #print(N)

# # if N % 2 == 1:
# #     print('홀수')
# # else:
# #     print('짝수')

# # N = int(input()) 

# # if N % 2 == 1:
# #     print('홀수')
# # else:
# #     print('짝수')

# TC = int(input())

# for i in range(TC):
#     N =  int(input())
#     #print(N)

#     if N % 2 == 1:
#         print('홀수')
#     else:
#         print('짝수')


# # 1차원 리스트 input 받기

# number = input().split()

# for number in numbers:
#     int_num = int(number)

#     if int_num % 2 == 1:
#         print(f'{int_num}은 홀수 입니다.')



# numbers = list(map(int, input().split()))

# for number in number:
#     if number % 2



N, M = map(int, input().split())
#4 5를 받아오는것

for i in range(N):
    numbers =  list(map(int, input().split()))
    matrix.append(numbers)

for row in matrix:
    #print(row)
    for item in row:
        if item == 10:
            print('5가 있습니다.')

N, M = map(int, input().split())
matrix = []

#아래 코드는 가로, 세로 순으로 도는 코드
for i range(N):
    numbers = list(map(int, input().split()))
    matrix.append(numbers)

for row in range(N):
    for col in range(N):
        print(matrix[row][col])

'''
위와 같은 코드 
위에는 미리 N, M을 지정해 두었기 때문에 
range에 N 값을 넣어도 되지만
만약에 지정을 해주지 않았다면 matrix에 [0]을 넣어주면 된다

for row in range(len(matrix)):
    for col in range(len(matrix[0])):
'''

#아래 코드는 세로, 가로 순으로 도는 코드
for col in range(len(matrix[0])):
    for row in range(len(matrix)):
        printprint(matrix[col][row])






