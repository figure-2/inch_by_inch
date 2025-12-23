import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)

T = int(input())
moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
def isWall(x,y):
    if x < 0 or y < 0 or y >= N or x >= N:
        return True
    return False
for tc in range(1, T+1):
    N = int(input())
    BRD = [list(map(int, input())) for _ in range(N)]
    x, y = 0, 0
    for i in range(N):
        if 2 in BRD[i]:
            x = BRD[i].index(2) 
            y = i 
            break
    Stack = [(y, x)]
    result = 0
    while Stack:
        y, x = Stack.pop()
        BRD[y][x] = 1
        for _y, _x in moves:
            dy = y + _y
            dx = x + _x
            if isWall(dy, dx):
                continue
            if BRD[dy][dx] == 3:
                result += 1
                break
            elif not BRD[dy][dx]:
                Stack.append((dy, dx))

        else:
            continue
        break
 
    print('#{} {}'.format(tc, result))
