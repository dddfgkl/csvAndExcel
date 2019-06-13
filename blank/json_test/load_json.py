import json
import sys
import os

def loadData():
    json_path = "c:/python/blank/json_test/small.json"
    json_file = open(json_path,mode='r',encoding='utf8')
    json_content = json.load(json_file)
    print(json_content,type(json_content))
    print(json_content['employees'])
    print(json_content['employees'][0])

    print(os.getcwd())

def main():
    loadData()
    print("main thread over")

main()