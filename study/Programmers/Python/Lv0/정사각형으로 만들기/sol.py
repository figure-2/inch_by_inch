# arr 
# 행의 수가 열보다 많다면 행의 끝에 0을 추가 row > column 
# 열의 수가 행보다 많다면 열의 끝에 0을 추가 row < column
# arr index 수가 행이고 그 행 안에 index가 열

arr = [[572, 22, 37], [287, 726, 384], [85, 137, 292], [487, 13, 876]]

def solution(arr):
    row = len(arr)
    col = len(arr[0])

    if row > col:
        for i in range(row):
            arr[i] += (row - col) * [0]
    elif row < col:
        for i in range(col - row):
            arr.append(col * [0])

    return arr


    
    for i in range(arr):
        for k in range(i):
            if i > k :
                i += 0
            else:
                k += 0


def solution(arr):
    num_rows = len(arr)
    num_cols = len(arr[0])

    if num_rows > num_cols:
        for i in range(num_rows):
            arr[i] += [0] * (num_rows - num_cols)
    elif num_cols > num_rows:
        for i in range(num_cols - num_rows):
            arr.append([0] * num_cols)

    return arr
    

def solution(arr):
    max_size = max(len(arr), len(arr[0]))
    answer = [[0]*max_size for _ in range(max_size)]

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            answer[i][j] = arr[i][j]

    return answer

