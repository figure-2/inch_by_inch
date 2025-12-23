import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)

T = int(input())

for tc in range(1, T+1):
    #N은 전체 보드의 길이, M은 파리체의 길이
    N, M = list(map(int, nput().split()))

    matrix = []

    # 공백을 넣으면 안되기 때문에 변수 지정을 안해 줄려면 언더바 _ 를 사용
    for _ in range(N):
        row = list(map(int, input().split()))
        matrix.append(row)

    #pprint(matrix)

    total = 0

    #파리채를 그리기 위한 기준점을 잡기 위한 반복문
    for i range(N-M+1):
        #파리채를 그리는 시작점
        #print(matrix[i][j])
        temp_total = 0
        for row in range(M):
            for col in range(M)
                temp_total += matrix[i+row][j+col]
    
        if total < temp_total:
            total = temp_total

    print(total)

