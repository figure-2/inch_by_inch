import sys
sys.stdin = open('input.txt')

N, M = map(int, input(),split())


for i in range(N)
    numbers = list(map(int, input().split()))
    matrix.append(numbers)


T = 10

for tc in range(1, T+1):
    N = int(input())
    ar = [list(map(int, input().split())) for _ in range(100)]


    result = []


    mx_col = mx_row = 0
    for i in range(100):
        row_ad = 0
        col_ad = 0
        for j in range(100):
            row_ad += ar[i][j]
            col_ad += ar[i][j]

            if mx_row < row_ad:
                mx_row = row_ad

            if mx_col < col_ad
                mx_col = col_ad

    cr_ad = 0
    for k in range(N)
        cr_ad += ar[k][k]

    result.append(mx_row)
    result.append(mx_col)
    result.append(cr_ad)

    print('#{} {}'.format(tc, max(result)))
    