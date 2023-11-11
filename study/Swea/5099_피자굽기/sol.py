import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)

for tc in range(1,int(input())+1):
 
    N, M = map(int, input().split())
    cheese = list(map(int,input().split())) # 치즈량 (//2 계산되기)
    # print(cheese)
    pizza_num = [i for i in range(M)] # 피자번호 0번부터
    # print(pizza_num)
    queue = pizza_num[0:N] # 피자번호로 돌리기(화덕큐) (삭제,삽입)
    # print(queue)
    while len(queue) != 1: # 마지막 남은게 한 개가 될 때까지
        if cheese[queue[0]] != 1:  # 치즈 0 으로 안 한 것은...1//2한게 0인지아닌지 
            cheese[queue[0]] = cheese[queue[0]] // 2 
            queue.append(queue.pop(0)) # 화덕큐 앞에 있는게 뽑아 뒤로 이동하기
        else: # 치즈량이 1이라면, //2 된게 => 어차피 0일테니, 바로 교체
            queue.pop(0)
            if N != M: # M개를 다 넣을 때까지 
                queue.append(pizza_num[N]) # pop한 자리에, 넣고 뒤로 돌린다 = 뒤로 바로 삽입 
                N += 1 # 다음넣을차례
 
 
    print('#{} {}'.format(tc,queue[0] +1)) # 피자번호 +1번