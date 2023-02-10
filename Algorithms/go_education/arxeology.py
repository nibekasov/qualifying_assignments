n = int(input())
s = list(map(int, input().split()))
a = list(map(int, input().split()))

ss1 = sum([[x] * y for x, y in zip(s, a)], [])

n = int(input())
s = list(map(int, input().split()))
a = list(map(int, input().split()))

ss2 = sum([[x] * y for x, y in zip(s, a)], [])

def subtract_lists(list1, list2):
    return [abs(x - y) * (i+1) for i, (x, y) in enumerate(zip(list1, list2))]

result = subtract_lists(ss1, ss2)

print(sum(result))
