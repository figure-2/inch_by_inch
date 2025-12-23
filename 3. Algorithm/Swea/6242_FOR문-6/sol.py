# bloods = ['A', 'A', 'A', 'O', 'B', 'B', 'O', 'AB', 'AB', 'O']
# bloods_dic = {'A': 0, 'O': 0, 'B': 0, 'AB': 0}

# for i in bloods:
#     if i == 'A':
#         bloods_dic['A'] += 1
#     elif i == 'O':
#         bloods_dic['O'] += 1
#     elif i == 'B':
#         bloods_dic['B'] += 1
#     else:
#         bloods_dic['AB'] += 1

# print(bloods_dic)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

# bloods_list = ['A', 'A', 'A', 'O', 'B', 'B', 'O', 'AB', 'AB', 'O']
# bloods_dict = {'A': 0, 'B': 0, 'O': 0, 'AB': 0}

# for blood in bloods_list:
#     bloods_dict[blood] += 1

# print(bloods_dict)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

# location = 'a'
# #location_list = 'b'
# #location_dict = 'c'

# location_list = ['서울', '부산', '서울', '서울', '대전', '제주', '광주', '부산', 'LA', 'NY']

# location_dict = {}

# for a in location_list:
#     # 이미 기록을 한 경우
#     if a in location_dict.keys():
#         location_dict[a] += 1
#     # 처음 기록되는 경우
#     else:
#         location_dict[a] = 1

# print(location_dict)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

location = 'a'
location_list = 'b'
location_dict = 'c'

b = ['서울', '부산', '서울', '서울', '대전', '제주', '광주', '부산', 'LA', 'NY']

c = {}

for a in b:
    # 이미 기록을 한 경우
    if a in c.keys():
        c[a] += 1
    # 처음 기록되는 경우
    else:
        c[a] = 1

print(c)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

a ='location'
b ='location_list'
c ='location_dict'

b = ['서울', '부산', '서울', '서울', '대전', '제주', '광주', '부산', 'LA', 'NY']

c = {}

for a in b:
    # 이미 기록을 한 경우
    if a in c.keys():
        c[a] += 1
    # 처음 기록되는 경우
    else:
        c[a] = 1

print(c)