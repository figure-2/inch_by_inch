# numer 분자/ denom 분모

    
def solution(numer1, denom1, numer2, denom2):
    up = numer1 * denom2 + numer2 * denom1
    down = denom1 * denom2  # 분수의 합 계산 

    max_num = max(up, down)
    max_value = 1
    
    for num in range(max_num, 0, -1):
        if down % num == 0 and up % num == 0:  # 0으로 나누었을때 약수인지 확인. 
            max_value = num
            break # 최대 공약수를 찾고 멈춤
    
    answer = [up / max_value, down / max_value]  # 최대 공약수를 이용해서 분자 분모를 나눈다.
    return answer