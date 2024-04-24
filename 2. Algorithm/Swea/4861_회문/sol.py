import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)


import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)


T = int(input())

for t in range(1, T + 1):
    N, M = map(int, input().split())  
    result = []
    #result2 = []

    numbers = []
    for i in range(N):
        numbers.append(input())

    for r in range(N):  
        for c in range(N - M + 1):  
            if numbers[r][c : c + M] == numbers[r][c : c + M][ : : -1]:
                result.append(numbers[r][c : c + M]) 


#전치 행렬 만들어보기

    for c in range(N):  
        for r in range(N - M + 1):  
            r_1 = []

            # if numbers[c][r : r + M] == numbers[c][r : r + M][ : : -1]:
            #     r_1.append(numbers[c][r : r + M]) 
             

    #print('#{} {}'.format(t, result[]))

