# -*- coding: UTF-8 -*-

from contextlib import contextmanager
import os
import time
import string

try:
    import MySQLdb

    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


def __virtual__():
    if not HAS_MYSQL:
        return False
    return 'mysql'


# def getConfigData(tag):
#     config_file = 'config.ini'
#     datalist = {}
#     if os.path.isfile(config_file) == False:
#         print("ERROR " + config_file + " file not found")
#         return datalist
#     config = ConfigParser.ConfigParser()
#     config.read(config_file)
#     options = config.options(tag)
#     for key in options:
#         context = config.get(tag,key)
#         datalist[key] = context
#     return datalist
mysql_conf = {'host':'xxx', 'user':'root',
                'port':'xxx', 'password':'xxx','db':'xxxx' }

@contextmanager
def _get_serv(mysql_conf, commit=False):
    '''
    Return a mysql cursor
    '''
    # mysql_conf = getConfigData('MYSQL')
    conn = MySQLdb.connect(host=mysql_conf['host'],
                           user=mysql_conf['user'],
                           port=int(mysql_conf['port']),
                           passwd=mysql_conf['password'],
                           charset='utf8')
    cursor = conn.cursor()
    conn.select_db(mysql_conf['db'])

    try:
        yield cursor
    except MySQLdb.DatabaseError as err:
        error = err.args
        print(error)
        cursor.execute("ROLLBACK")
        raise err
    else:
        if commit:
            cursor.execute("COMMIT")
        else:
            cursor.execute("ROLLBACK")
    finally:
        conn.close()


def _get_format_now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# 这个不用看，暂时没有update数据库的需求
def update_by_params(mysql_conf, record, params, table):
    return
    sal = ''
    if (len(record) > 0):
        sql = "UPDATE " + table + " set "
        set_values = []
        for k, v in record.iteritems():
            set_values.append(str(k) + "='" + str(v) + "'")
        sql = sql + string.join(set_values, ',')
    else:
        return False
    if (len(params) > 0):
        where = []
        for k, v in params.iteritems():
            where.append(str(k) + "='" + str(v) + "'")
        sql = sql + " WHERE " + string.join(where, ' AND ')
    with _get_serv(mysql_conf, commit=True) as cur:
        cur.execute(sql)
        return cur.fetchall()

# record 字段 , params 是字典形式where限定, table就是table名字
def get_info_by_params(mysql_conf, record, params, table):
    sql = ''
    if (len(record) > 0):
        sql = "SELECT " + ','.join(record) + " FROM " + table
    else:
        sql = "SELECT * FROM " + table
    if (len(params) > 0):
        where = []
        for k in params:
            val = params[k]
            if isinstance(val, str):
                where.append("%s=\"%s\"" % (k, val))
            elif isinstance(val, int):
                where.append("%s=%d" % (k, val))
            else:
                print("key:%s,type:%s" % (k, type(val)))

        sql = sql + " WHERE " + ' AND '.join(where)
    # print("sql:",sql)
    with _get_serv(mysql_conf, commit=True) as cur:
        cur.execute(sql)
        return cur.fetchall()


def execute_sql(mysql_conf, sql):
    with _get_serv(mysql_conf, commit=True) as cur:
        cur.execute(sql)
        return cur.fetchall()


def insert_many(mysql_conf, sql, tuple_list):
    with _get_serv(mysql_conf, commit=True) as cur:
        affected_rows = cur.executemany(sql, tuple_list)
        return affected_rows

# 注意一下cursor.execute返回的是一个tuple,tuple中的元素还是一个tuple，这样符合database的特点
def main():
    sql = 'select sub_category from amap_type limit 10'
    retList = execute_sql(mysql_conf, sql)
    print(type(retList), len(retList))
    for i in range(len(retList)):
        print(retList[i][0], type(retList[i][0]),type(retList[i]))
    print('main thread over')

if __name__ == "__main__":
    main()