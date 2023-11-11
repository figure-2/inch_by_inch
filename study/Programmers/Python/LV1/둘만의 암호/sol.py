def solution(s, skip, index):
    answer = ''
    
    for c in s:
        i = ord(c)
        j = index
        while j > 0:
            i += 1
            if i > ord('z'):
                i = ord('a')
            if chr(i) in skip:
                j += 1
            j -= 1
        answer += chr(i)
    
    return answer


from string import ascii_lowercase

def solution(s, skip, index): # use index instead of n
    result = ''
    apb_list = list(ascii_lowercase)
    
    for i in skip:
        apb_list.remove(i)

    for j in s:
        if apb_list.index(j) + index > len(apb_list)-1: # use index instead of n
            result += apb_list[apb_list.index(j) + index-(len(apb_list))] # use index instead of n
        else:
            result += apb_list[apb_list.index(j) + index] # use index instead of n

    return result






from string import ascii_lowercase

def solution(s, skip, index):
    result = ''
    apb_list = list(ascii_lowercase)
    
    for i in skip:
        apb_list.remove(i)

    for j in s:
        try:
            new_index = (apb_list.index(j) + index) % len(apb_list)
            result += apb_list[new_index]
        except ValueError: # Catch the specific exception.
            result += j # Or handle it in a way that fits the problem's requirements.

    return result

