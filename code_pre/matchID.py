# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from os import path
from code_pre import my_modules

# 将前面两步：preprocess_label.py和preprocess_text.py得到的结果读取进来，


label = pd.read_table(path.join(path.dirname(__file__),'..','data','labels_trans2num.txt'),
                      header=None, sep=',', encoding='utf-8')
print(label)

text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','cn_text_pd'))
singleText = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','cn_singleText_pd'))

# num_clusters = 21
# text_emoji = pd.read_pickle('../np_data/text_emoji_'+str(num_clusters)+'_pd')

# print(text)
# print(singleText)

def find_text(id):
    id_index = 0
    find_flag = 0
    for i in range(len(text)):
        if text[id_index][i] == id:
            find_flag = 1
            return text['text_seg'][i]
        else:
            continue
    if find_flag == 0:
        print('text_seg no found')
        exit(1)

def find_singleText(id):
    id_index = 0
    list = []
    for i in range(len(singleText)):
        if singleText[id_index][i] == id:
            list.append(singleText['singleText_seg'][i])
        else:
            continue
    if len(list) == 0:
        print('singleText_seg no found')
        exit(1)
    else:
        return list

# def find_emoji(id):
#     id_index = 0
#     find_flag = 0
#     for i in range(len(text_emoji)):
#         if text_emoji[id_index][i] == id:
#             find_flag = 1
#             return text_emoji['emoji_fea'][i]
#         else:
#             continue
#     if find_flag == 0:
#         print('text_seg no found')
#         exit(1)

label['text_all'] = label[0].apply(lambda id: find_text(id))
label['text_single'] = label[0].apply(lambda id: find_singleText(id))
# label['emoji_fea'] = label[0].apply(lambda id:find_emoji(id))
# print(label['emoji_fea'])

label.to_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_pd'))
# arr = np.array(list(label['emoji_fea']))
# np.save('../np_data/vectors_emoji_'+str(num_clusters)+'.npy', arr)