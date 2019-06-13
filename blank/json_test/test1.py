import numpy as np 
import os
import re

def load1():
    print("load1")
    dict1 = {"aaa","bbb","ccc"}
    for item in dict1:
        print(item)
    
    myst = 'a'.join("xxxq")
    myre = re.compile(myst)

    print(myre,type(myre))
    myre.scanner()


def load2():
    print("load2")
    absPath = os.path.abspath(__file__)
    dirname = os.path.dirname(absPath)
    print(absPath)
    print(dirname)
    PROJECT_ROOT = os.path.join(dirname, '../')
    print(PROJECT_ROOT)


def main():
    load2()

main()
