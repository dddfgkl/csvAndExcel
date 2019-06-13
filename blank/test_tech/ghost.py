def f():
    x = ''
    y = lambda x : x.strip()
    print(y(x), 'what')
    if y(x):
        print('false')
    else:
        print('right')
    # 在python中空是有特定意义的
    if x:
        print('xxxx')
    else:
        print('yyyy')

if __name__ == "__main__":
    f()