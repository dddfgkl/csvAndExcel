import os 
import csv 
import pandas as pd 
import numpy as np 
"""
desc: 分析两个csv文件的异同，旧模型测试脚本生成的csv文件和新模型测试脚本生成的csv文件
"""
def load_data():
    firstFile_path =""
    secondFile_path =""
    resultFile_path = ""
    if os.path.exists(firstFile_path) and os.path.exists(secondFile_path) and os.path.exists(resultFile_path):
        print("file exist, the process can continue")
    else:
        print("file not exist")
        return
    first_file = pd.read_csv(firstFile_path, engine="python")
    second_file = pd.read_csv(secondFile_path, engine="python")
    result_file = open(resultFile_path, mode='a', newline='')
    m1, n1 = first_file.shape
    m2, n2 = second_file.shape
    print("first file : {0}行，{1}列".format(m1, n1))
    print("first file : {0}行，{1}列".format(m2, n2))

    '''
    print(first_file.columns)
    print(first_file.loc[18])
    print(first_file.loc[18].equals(second_file.loc[18]))
    '''

    result_writer = csv.writer(result_file)
    print()
    #result_writer.writerow(first_file.loc[0])
    def_row = 2997
    keyIn = input("make sure your self def_row is right, continue(yes/no?)")
    if keyIn != "yes":
        return
    for i in range(def_row):
        print(first_file.loc[i][0])
        if first_file.loc[i][5] != second_file.loc[i][5]:
            k = []
            for j in range(n1):
                k.append(first_file.loc[i][j])
                k.append(second_file.loc[i][j])
            result_writer.writerow(k)
    
    result_file.close()

def unit_test():
    load_data()

def main():
    unit_test()

if __name__ == "__main__":
    main()