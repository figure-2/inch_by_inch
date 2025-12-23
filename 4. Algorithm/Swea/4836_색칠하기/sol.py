import sys
sys.stdin = open('input.txt')

for t in range(int(input())):
   
    red_lst = []
    blu_lst = []
    count = 0
    
    for _ in range(int(input())):
       
        r1, c1, r2, c2, color = map(int, input().split())
  
        for x in range(r1, r2+1):
            for y in range(c1, c2+1):
                if color == 1:
                    red_lst.append([x,y])
 
                if color == 2:
                    blu_lst.append([x,y])
                
    for common in red_lst:
        count += blu_lst.count(common)
   
    print(f'#{t+1} {count}')