import pandas as pd 
import os 


def load_name_form_csv():
    name_path = ""
    if os.path.exists(name_path) == False:
        print('file not exist')
        return
    name_file = pd.read_csv(name_path, engine="python", header=None)
    file_list = []
    m, n = name_file.shape
    for i in range(m):
        k = name_file.loc[i][0]
        file_list.append(k)
    return file_list

def database_table_list_unit_test():
    database_tabelName = load_name_form_csv()
    for name in database_tabelName:
        print(name)

if __name__ == "__main__":
    database_table_list_unit_test()