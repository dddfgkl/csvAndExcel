import os 
import csv 
import time
import pandas as pd 
import numpy as np 
"""
desc1: 数据清洗脚本
desc: 数据去重，选取具有代表性的pio_type，原始数据是从数据库中直接拉去下来的数据，同类型的数据出现频率太高
      这个脚本的作用是清洗数据，选择部分数据作为训练集或者测试集
demand: 数据是csv格式的文件，最好在csv文件中已经去重
"""
def load_data():
    # 设置读取文件路径和存储文件路径
    file_path = ""
    result_path = ""
    if os.path.exists(file_path) and os.path.exists(result_path):
        print("file exist, the process can continue")
    else:
        print("file not exist")
        return
    raw_file = pd.read_csv(file_path, engine="python")
    result_file = open(result_path, mode='a', newline='')

    # 交互性命令行设置，方便设置抽取参数sampel_num
    m, n = raw_file.shape
    result_writer = csv.writer(result_file)
    sampel_num = input("how many data do you want extract from the origin file :")
    if sampel_num == None:
        print("we set a default number for you, number is 10000")
        sampel_num = 10000
    sampel_num = int(sampel_num)
    if sampel_num >= m:
        print("invalid number")
    keyIn = input("make sure you have define how many data do you want, continue(yes/no?)")
    if keyIn != "yes":
        return
    print("the process start working")
    
    # 核心代码，进行数据数据
    begin_time = time.clock()
    print(0, raw_file.loc[0][1])
    print(1, raw_file.loc[1][1])
    for i in range(m):
        # k = raw_file.loc[i][1]
        # print(k, type(k))
        # 注意这里raw_file.loc[i][1]的type是str类型,只比较字符串的第一个字符来判断是否选取这个样本
        if i >= sampel_num:
            break
        if i == 0 or raw_file.loc[i][1][0] != raw_file.loc[i-1][1][0]:
            # print(raw_file.loc[i][1])
            kk = raw_file.loc[i][1]
            print(kk)
            result_writer.writerow([kk])
    print("data storing...")
    result_file.close()
    end_time = time.clock()
    print("time consuming of the process:", end_time-begin_time, "s")

def main():
    load_data()

if __name__ == "__main__":
    main()