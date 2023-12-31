import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)


T = int(input())
for tc in range(1, T+1):
    n,m = map(int, input().split())
    nums = list(map(int, input().split()))

    for _ in range(m):
        nums.append(nums.pop(0))
        
    print(f'#{tc} {nums[0]}')