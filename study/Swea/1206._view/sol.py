import sys
sys.stdin = open('input.txt')

for test_case in range(1,11):
    result = 0
    houseCount = int(input())
    house = list(map(int , input().split()))
    for i in range(2, houseCount-2):
        def_2 = house[i] - house[i-2]
        def_1 = house[i] - house[i-1]
        def1 = house[i] - house[i+1]
        def2 = house[i] - house[i+2]
        if def_2 > 0 and def_1 > 0 and def1 > 0 and def2 > 0 :
            result += min(def_2, def_1, def1, def2)
 
    print("#{} {}".format(test_case,result))


for test_case in range(1,11):
    result = 0
    houseCount = int(input())
    house = list(map(int , input().split()))
    for i in range(2, houseCount-2):
        arMax = max(house[i-1],house[i-2],house[i+1],house[i+2])
        if house[i] > arMax:
            result += ( house[i] - arMax )
 
    print("#{} {}".format(test_case,result))


T = 10
for test_case in range(1, T + 1):
    N = int(input())
    lst = list(map(int, input().split()))
    ans = 0
    for i in range(2, N-2):
        mx = max(lst[i-2:i]+lst[i+1:i+3]) # 4개 값을 직접 써줘도 됨
        if lst[i] > mx:
            ans += lst[i]-mx
    print(f'#{test_case} {ans}')


    

T = 10
for test_case in range(1, T + 1):
    N = int(input())
    lst = list(map(int, input().split()))
    #빌딩들의 높이가 저장되어있다
    ans = 0
    for i in range(2, N-2):
        mx = lst[i-2]
        for j in range(i-1, i+3):
            if i == j:
                continue
            if mx < lst[j]:
                mx = lst[j]
        if lst[i] > mx:
            ans += lst[i]-mx
    print(f'#{test_case} {ans}')