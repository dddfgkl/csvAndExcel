import tensorflow as tf 
import numpy as np 
import json
import os
from json import JSONDecodeError

DATA_ROOT_FOLDER = "./data/"

TEMPLATE_LARGE = "template_large"
TEMPLATE_SMALL = "template_small"

# entity文件每行中元素的分隔符
ENTITY_LINE_SEPARATOR = ";|\t"

# TEMPLATE = [TEMPLATE_LARGE, TEMPLATE_SMALL]

# 关键词等少量可以枚举的实体放在entity.json中
ENTITY = "entity"
# 实体和对应的实体文件路径放在entity_data.json中
ENTITY_DATA = "entity_data"
UTT_COUNT = "utterance_count"

CONFIG_ITEMS = [TEMPLATE_LARGE, TEMPLATE_SMALL, ENTITY, ENTITY_DATA, UTT_COUNT]
DATA_ITEMS = [TEMPLATE_LARGE, TEMPLATE_SMALL, ENTITY, UTT_COUNT]

class ConfigLoader():
    def __init__(self):
        self.configs = {}
        self.load(CONFIG_ITEMS)

    def load(self, config_items):
        for item in config_items:
            # print (item)
            with open("./config/" + item + ".json", mode='r', encoding='utf8') as f:
                try:
                    self.configs[item] = json.load(f)
                    # f是加载的json字典数据，configs还是一个字典
                except JSONDecodeError as err:
                    print("JSON error: {0}".format(err))
                    print ("./config/" + item + ".json")
                    exit(1)

class DataLoader():
    def __init__(self, root_folder, config):
        self.root_folder = root_folder
        self.config = config
        self.data = {}
        self.district = District()
        self.district.load_district(os.path.join(root_folder, 'district'))

        for item in DATA_ITEMS:
            if item in CONFIG_ITEMS:
                self.data[item] = config.configs[item]

    def __load_entity_data(self, entity_type):
        entity_data = self.config.configs[ENTITY_DATA]
        if entity_type in entity_data:
            filePath = os.path.join(self.root_folder,entity_data[entity_type])
            # print ("Load " + entity_type + " from: " +filePath)
            with open(filePath, mode='r', encoding='utf8') as f:
                entity_items = set()
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    items = re.split(ENTITY_LINE_SEPARATOR, line)
                    for item in items:
                        entity_items.add(item)
            self.data[ENTITY][entity_type] = list(entity_items)
        else:
            print("LOAD DATA FAILED! " + entity_type)
            assert (0)

    def random_sample(self, type, value):
        if type == ENTITY and is_district_type(value):
            # 生成省市县数据
            item = self.district.get_full_district()
        else:
            if type in self.data and value in self.data[type]:
                    item = choice(self.data[type][value])
            else:
                # 因为entity数据文件较大，所以在使用时才加载
                assert (type == ENTITY)
                self.__load_entity_data(value)
                item = choice(self.data[type][value])

        return item

class TemplateList:
    def __init__(self):
        self.template_list = {}

    def add_template(self, intent, templateHolder):
        if intent in self.template_list:
            self.template_list[intent].append(templateHolder)
        else:
            self.template_list[intent] = [templateHolder]

    def random_sample_template(self, intent):
        return choice(self.template_list[intent])


def load_template(templateList, data_loader, template_type=TEMPLATE_LARGE):
    for intent in data_loader.data[template_type]:
        for template_string in data_loader.data[template_type][intent]:
            templateHolder = TemplateHolder()
            templateHolder.extract(intent, template_string)
            templateList.add_template(intent, templateHolder)

# 用来存储一个模板
# 可以从模板字符串中提取一个模板
# 可以用该模板生成句子
# 注意这个是个类
class TemplateHolder:
    def __init__(self):
        self.template = []

    # 从模板字符串中提取模板
    def extract(self, intent, template_string):
        self.intent = intent
        expr_extractor = ExpressionExtractor(template_string)
        for expr in expr_extractor.extract():
            self.template.append(expr)

    # 用该模板生成句子
    def generate_utt(self, data_loader):
        plain_text = ""
        slot_text = ""
        index = 0

        # 以一定的概率在句首和句尾加入闲聊
        head_chat = random_get_entity(data_loader, "chitchat", 0.08)
        tail_chat = random_get_entity(data_loader, "chitchat", 0.08)

        for expr in self.template:
            if expr.type == TYPE_TEXT:
                plain_text += expr.value
                slot_text += expr.value
            else:
                text = data_loader.random_sample(expr.type.lower(), expr.value)

                # 如果句子中的第一个词为"kw_search"，则以60%的概率在它的前面加上助词，
                # 比如“帮我”，“我要”等
                if (index == 0 and
                        expr.type.lower() == ENTITY and
                        expr.value == "kw_search" and
                        text != "哪里有"):
                    if random.uniform(0,1) <= 0.6:
                        auxiliary = data_loader.random_sample(expr.type.lower(), "kw_auxiliary")
                        text = auxiliary + text

                # 以一定的概率加入口语词
                modal_word = random_get_entity(data_loader, "modal_word", 0.1)

                if text != None:
                    # 以一定的概率从字符串中删除一个字
                    text = random_remove_char(text, 0.08)
                    plain_text += text
                    plain_text += modal_word
                text = self.add_tag(expr.type, expr.value, text)
                slot_text += text
                slot_text += modal_word
            index += 1
        return head_chat + plain_text + tail_chat, head_chat + slot_text + tail_chat

    def add_tag(self, type, value, text):
        startTag = "<" + TAG_MAP[type] + ":" + value + ">"
        endTag = "</" + TAG_MAP[type] + ":" + value + ">"
        if text == None:
            text = "n/a"
        return startTag + text + endTag


# 这个是主函数，templateHolder也有一个方法叫做generate_utt，注意他们两个是不一样的
def generate_utt(outfile, times=1.0):
    # load config and data
    print("Load config")
    config = ConfigLoader()
    print("Load data")

    data_loader = DataLoader(__DATA_ROOT_FOLDER__, config)

    # load template
    print("Load template")
    templateList_large = TemplateList()
    templateList_small = TemplateList()
    load_template(templateList_large, data_loader, TEMPLATE_LARGE)
    load_template(templateList_small, data_loader, TEMPLATE_SMALL)

    writer = open(outfile, mode='w', encoding='utf8')

    print("Start generate utterance")
    for intent, count in data_loader.data[UTT_COUNT].items():
        for i in range(int(int(count[0])*times)):
            templateHolder = templateList_large.random_sample_template(intent)
            plain_text, slot_text = templateHolder.generate_utt(data_loader)
            writer.write(plain_text + "\t" + intent + "\t" + slot_text + "\n")

        for i in range(int(int(count[1])*times)):
            templateHolder = templateList_small.random_sample_template(intent)
            plain_text, slot_text = templateHolder.generate_utt(    )
            writer.write(plain_text + "\t" + intent + "\t" + slot_text + "\n")

    print("Done generate utterance")

def main():
    #test1()
    print("main over")

main()

