def sum_of_squares(numbers:list)->int:
    #Get list and return sum of squares
    return sum([x**2 for x in numbers])

def min_max_square(n:int, k:int)->tuple:
    '''
    Solution of problem:
    Return minimum and maximum square 
    having n*4 pieces of fence and
    k paddock 
    '''
    max_size = (n-k+1)**2 + k-1
    min_size = 0
    i = 0
    while n > k*i:
        i += 1

    i -= 1
    ss = [i] * k
    t = n - k*i
    for j in range(t):
        ss[j] += 1

    min_size = sum_of_squares(ss)

    return min_size, max_size

n = int(input())
k = int(input())

ans = min_max_square(n,k)
print(ans[0], ans[1)
