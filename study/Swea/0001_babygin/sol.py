import sys
sys.stdin = open('input.txt')

T = int(input())

for tc in range(1, T+1):
    numbers = (input())  
    #int(input())로 하면 맨 앞에 0을 못 읽기 떄문에 그냥 (input())으로
    counter = [0 for _ in range(10)]
    # 0이 10개가 차있는 리스트가 만들어진다

    for number in numbers:

        counter[int(number)] += 1
        # 리스트에 0 ~ 10번쨰 자리에 각각 몇개씩 들어가있는지 확인하기 위한 코드
    #print(numbers)

    idx = 0
    # 그동안은 리스트 자체에 접속을 했지만 index접근을 통햇 확인한다
    is_babygin = 0
    
    while idx < len(counter):

        # 1. triplet 인지 검증
        if counter[idx] > 3:
            counter[idx] -= 3 # 재사용하면 안되기 때문에 3을 뺴준다, 3을 뺴고 0으로 채워준다
            is_babygin += 1 # 베이비진에 1이 들어있으면

        # 2. run 인지 검증
        # 나를 기준으로 오른쪽 2칸을 봐야함. 만약 인덱스 범위가 넘어가는곳을 봐야 한다면 에러가 뜸
        if idx < len(counter) - 2:
        # 카운터 (10)에서 적은 8까지 run을 돌리라는 것
            if counter[idx] and counter[idx+1] and counter[idx+2]:
            # and 연산자는 false가 있으면 다 false임, 만약 1, 0, 2 면 false
                is_babygin += 1
                counter[idx] -= 1
                counter[idx+1] -= 1
                counter[idx+2] -= 1
                #카드를 한장씩 섰기 때문에 1개씩 뺴준다        
        
        idx += 1

    if is_babygin == 2:
        print(f'#{tc} Ture')
    
    else:
        print(f'#{tc} False')