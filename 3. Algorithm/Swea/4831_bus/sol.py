import sys
sys.stdin = open('input.txt')

T = int(input())

for tc in range(1, T+1):

    K, N, M = map(int, input().split())
    charge_spot = list(map(int, input().split()))

    now = 0
    cnt = 0
  
    while now + K < N:
        for i in range(K, 0, -1):
            
            if (now + i) in charge_spot:
                
                now += i
                cnt += 1
                break
            

        else: 
            cnt = 0
            break

    print("#{} {}".format(tc, cnt))