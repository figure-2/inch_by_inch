# 첫글자 x // x 와 x가 아닌 다른 글ㅈ


def solution(s):
    answer = 0  # 분해한 문자열의 개수
    a1, b1 = 0, 0  # x와 같은 글자 수, 다른 글자 수
    for i in range(len(s)):
        if a1 == b1:  # 두 횟수가 같으면 분리(answer+1)
            answer += 1
            x = s[i]
            a1, b1 = 0, 0
            
        if s[i] == x:
            a1 += 1
        else:
            b1 += 1
            
    return answer


    # for i in range(len(s)):
    #     if isx == isnotx:  # 두 횟수가 같으면 분리(answer+1)
    #         answer += 1
    #         x = s[i]
    #         isx, isnotx = 0, 0
                
    #     if s[i] == x:
    #         isx += 1
    #     else:
    #         isnotx += 1



# https://dduniverse.tistory.com/entry/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4-%EB%AC%B8%EC%9E%90%EC%97%B4-%EB%82%98%EB%88%84%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%AC-python