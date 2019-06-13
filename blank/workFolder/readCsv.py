import os 
import csv
import numpy as np 
import pandas as pd 

file_path = "E:/sunday/readme.csv"
def load_dataFromCsv():
    if os.path.exists(file_path):
        print("path file exit")
    else:
        print("path file not exit")
        return
    # 读取失败有可能是csv格式不对
    csv_data = pd.read_csv("E:/sunday/readme.csv", engine="python", header=None)
    print(csv_data, type(csv_data))
    m, n = csv_data.shape
    print("{0}行，{1}列".format(m, n))
    print()
    for i in range(m):
        for j in range(n):
            print(csv_data.iloc[i,j], type(csv_data.iloc[i,j]))
    print("load_data over!")

def csv_mm():
    legal_path = "E:/sunday/readme.csv"
    if os.path.exists(legal_path):
        print("path file exit")
    else:
        print("path file not exit")
        return
    open_csvData = open(legal_path, mode='a',newline='')
    out_csv = csv.writer(open_csvData)
    ner_rows = [9,8,7]
    out_csv.writerow(ner_rows)
    open_csvData.close()

def unit_test():
    csv_mm()

def main():
    unit_test()

if __name__ == "__main__":
    main()
