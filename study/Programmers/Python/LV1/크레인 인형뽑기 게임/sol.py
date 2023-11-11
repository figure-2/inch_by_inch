# board 가 행
# moves 크래인이 움직이는 순서

def solution(board, moves):
    answer = 0
    doll_list = [] # 바구니
    # 순서대로 크레인의 움직임 위치를 받는다.
    for i in moves:
        for j in range(len(board)):
            # 뽑힐 인형이 없다면 다음 동작으로 넘어감
            if board[j][i-1] == 0:
                continue
            else:
                doll_list.append(board[j][i-1])
                board[j][i-1] = 0
                # 인형이 두개이상 쌓였을 때
                if len(doll_list) > 1:
                    # 끝에 있는 두개가 동일하면
                    if doll_list[-1] == doll_list[-2]:
                        # 인형이 터지니까 +2
                        doll_list.pop(-1)
                        doll_list.pop(-1)
                        answer += 2
                break
    return answer

# row 가로 col 세로

def solution(board, moves):
    answer = 0
    bucket = []
    
    for row in moves: # 순서대로 크레인의 움직일 위치
        for col in board: 
            if col[row-1] != 0:  # 실제 위치를 위해서 -1
                bucket.append(col[row-1])
                col[row-1] = 0
                break

        if len(bucket) >= 2 and bucket[-1] == bucket[-2]: #인형이 두개 이상 쌓였을때. 맨 뒤에 있는 두개 가 같타면 
            answer += 2
            bucket = bucket[:-2]
            
    return answer