import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)


T = int(input())

for tc in range(1, T+1):
    # V : 노드 수 / E : 간선 수
    V, E = list(map(int, input().split()))

    nodes = [ [0] * V+1 for _ in range(V+1) ]
    #print(nodes)

    # 간선의 갯수만큼 반복을 진행
    for line in range(E):
        start, end = list(map(int, input().split()))
        nodes[start][end] = 1
    #pprint(nodes)

    # S : 출발노드 / G : 도착노드
    S, G = list(map(int,input().split()))
