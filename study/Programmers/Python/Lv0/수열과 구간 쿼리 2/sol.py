# 정수 arr 2차원 배열 queries //
# queries 의 원소는 각각 query [s, e, k]

def solution(arr, queries):
    answer = []

    for s, e, k in queries:
        answer_1 = []
        for i in arr[s:e+1]:
            if i > k:
                answer_1.append(i)
                
        if not answer_1:  # temp 리스트가 비었는지 확인
            answer.append(-1)
        else:
            answer.append(min(answer_1))  # 최솟값 찾아서 추가

    return answer


# def solution(arr, queries):
#     result = []
#     for query in queries:
#         temp_list = []
#         for i in range(query[0], query[1] + 1):
#             if arr[i] > query[2]:
#                 temp_list.append(arr[i])
#         try:   
#             result.append(min(temp_list))
#         except:
#             result.append(-1)
#     return result


# def solution(arr, queries):
#     answer = []
#     for s, e, k in queries:
#         tmp = []
#         for x in arr[s:e+1]:
#             if x > k:
#                 tmp.append(x)
#         answer.append(-1 if not tmp else min(tmp))
#     return answer