import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)

tc=int(input())

def f(N):
    if N%10==0:
        if N==10: 
            return 1
        elif N==20:
            return 3
        else:
            return f(N-10)+(2*f(N-20))
    else:
        print("10의 배수만 입력하세요")

for t in range(1, tc+1):
    N=int(input())
    count=f(N)
    print("#{} {}".format(t,count))





#강사님 풀의

T = int(input())

memo = [0, 1, 3]

for tc in range(1, T+1):
    N = int(input()) // 10

    # memo배열에 출력시킬 값이 없으면 추가
    while N >= len(memo): # N은 최대 몇칸을 만들지에 대한 정보
        # n-2  배열에 가로로 작은 사각형을 쌓거나 큰 사각형 쌓는 방법 ( x2 )
        # n-1  배열에 세로로 작은 사각형 쌓는 방법 하나.
        temp = 2 * memo[len(emo)-2] + memo[len(memo)-2]
        memo.append(temp)

    print(memo[n])


