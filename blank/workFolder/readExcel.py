import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

def load_data():
    file_path = "C:/python/blank/workFolder/test.xlsx"
    rowData = pd.read_excel(file_path,sheet_name=0, header=0)
    df = pd.DataFrame(rowData)
    print(type(df))
    
    # 存储修改后的excel
    df.to_excel(file_path, sheet_name=0, header=True)




def unit_test():
    load_data()

def main():
    unit_test()

if __name__ == "__main__":
    main()
    print("main thread over!")