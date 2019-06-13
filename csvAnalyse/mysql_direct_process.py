from contextlib import contextmanager
import os
import time
import string
import pandas as pf
import numpy as np 
import csv 
from mysql_wrapper import *
from database_table_nameList import load_name_form_csv

tableName_list = load_name_form_csv()
global_number = 0

def load_from_mysql(sql):
    if sql == None:
        return 
    #print('make sure you have set sql sequeue, continue(yes/no)?')
    global global_number
    if global_number == 0:
        can_continue = input('make sure you have set sql sequeue, continue(yes/no)?')
    else:
        can_continue = 'yes'
    if can_continue != 'yes':
        print('program terminate')
        return
    # sql = 'select sub_category from amap_type limit 10'
    retList = execute_sql(mysql_conf, sql)
    '''
    print(type(retList), len(retList))
    for i in range(len(retList)):
        print(retList[i][0], type(retList[i][0]),type(retList[i]))
    print('main thread over')
    '''
    print('load from database over, next step...')
    return retList

def process_from_mysql(sql):
    if sql == None:
        return 
    global global_number
    raw_data = load_from_mysql(sql)
    if raw_data == None:
        return 
    print('make sure you have set the result path')
    # 在这里设置输出文件路径
    result_path = ""
    if os.path.exists(result_path):
        print("file exist, the process can continue")
    else:
        print('invalid result path')
        result_file = open(result_path, mode='w', newline='', encoding='uft-8')
        result_file.close()
        return 
    if os.path.exists(result_path) == False:
        print("failed to construct a new file")
        return 
    result_file = open(result_path, mode='a', newline='', encoding='utf-8')
    result_writer = csv.writer(result_file)
    print('start execute', sql)
    begin_time = time.clock()
    for i in range(len(raw_data)):
        if i == 0 or raw_data[i][0][0] != raw_data[i-1][0][0]:
            global_number += 1
            #strip_str = raw_data[i][0.strip()
            result_writer.writerow(raw_data[i])
            print(raw_data[i])
    print("data storing...")
    result_file.close()
    end_time = time.clock()
    print("time consuming of the process:", end_time-begin_time, "s")

def extract_from_database():
    for table_name in tableName_list:
        table_name = table_name.strip()
        m_sql = 'select value from ' + table_name + ' limit 50000'
        process_from_mysql(sql=m_sql)
    print(global_number)
    print('data process over !!!!!!!!!!!')


if __name__ == "__main__":
    extract_from_database()

