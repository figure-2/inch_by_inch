memo = [0, 1]

def fibo(n):
    if n >= 2 and len(meno) <= n: # 0과 1이 있으니깐 2부터 구하는 과정
        meno.append(fibo(n-1) + fibo(n-2))
        return meno[n]

print(fibo(100))