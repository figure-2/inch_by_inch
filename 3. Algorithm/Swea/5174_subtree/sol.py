import sys
from pathlib import Path

file_path = Path(__file__).parent
input_path = file_path / 'input.txt'
sys.stdin = open(input_path)

def subtree(root):
    global cnt
    cnt += 1

    for i in tree[int(root)]:
        if i:
            subtree(i)

    return cnt


for tc in range(int(input())):
    E, N = map(int, input().split())

    nodes = list(input().split())
    tree = [[] for _ in range(E + 2)]
    for t in range(0, len(nodes), 2):
        tree[int(nodes[t])].append(nodes[t + 1])
    cnt = 0
    
    print("#%d %d" % (tc + 1, subtree(N)))