def log(x):
    """
    밑이 e인 자연로그
    입력값 : 진수 x
    출력값 : ln(x)
    """

    n = 100000000.0
    res = n*((x**(1/n))-1)
    print(res)
    return res

if __name__ == '__main__':
    log(10)