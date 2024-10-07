def factorial(x):
    x_list = list(range(1, x+1))
    res = 1
    for val in x_list:
        res *= val
    return res

def combination(n, x):
    """
    조합
    입력값 : n, x
    출력값 : nCx(실수)
    """
    res = factorial(n)/(factorial(x)*factorial(n-x))
    print(res)
    return res

if __name__ =='__main__':
    combination(5, 3)
