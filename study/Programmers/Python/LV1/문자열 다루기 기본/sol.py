def solution(s):
    if len(s) == 4 or len(s) == 6:
        try:      
            int(s)
            return True
        except:
            return False
    else:
        return True
        
print(solution('a234'))
print(solution('1234'))


def solution(s):
    if len(s) == 4 or len(s) == 6 :
        for i in s :
            if i >='0' and i<= '9' :
                continue
            else : 
                return False
        return True
    else :
        return False