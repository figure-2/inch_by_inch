import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)

def dfs(n, sum): 
    global ans
    if ans<=sum:  
# 최소 값을 만드는데 정답 보다 이미 크다면 같다면 음수가 없기 때문에 작아질 일이 없으면 
# 더 가봐도 갱신할 일이 없으니 
        return
 
    if n==N:
        ans = min(ans, sum)
        return
 
    for j in range(N):
        if v[j]==0:     # 사용하지 않은 숫자
            v[j]=1  # 이제 사용한 숫자를 1로 변경
            dfs(n+1, sum+arr[n][j]) # 백트래킹 다음 행으로 이동 .
            v[j]=0      # 잊지마세요!!!!
 
T = int(input())
for test_case in range(1, T + 1):
    N = int(input())
    arr = list(list(map(int, input().split())) for _ in range(N))
 
    ans = N*10  # ans 랑 비교하는 ans는 최소 값이니깐  n*10 혹은 100
    v = [0]*(N) 
    dfs(0, 0) #
 
    print(f'#{test_case} {ans}')