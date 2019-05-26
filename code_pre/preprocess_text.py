# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from os import path
from code_pre.my_modules import extract_cn


##数据清洗
## 输入：text_pd, singleText_pd
## 输出：cn_text_pd, cn_singleText_pd


text = pd.read_pickle(path.join(path.dirname(__file__), '..', 'np_data','text_pd'))
singleText = pd.read_pickle(path.join(path.dirname(__file__), '..', 'np_data', 'singleText_pd'))

# print(text) # text是index，id，原集合文本，分词之后的文本（这里的seg不要了，）
# print(singleText) # singleText是index，原单个文本，分词之后的文本

text['text_seg'] = text['text'].apply(lambda s: extract_cn(s))
singleText['singleText_seg'] = singleText['singleText'].apply(lambda s: extract_cn(s))

pd.to_pickle(text, path.join(path.dirname(__file__), '..', 'np_data','cn_text_pd'))
pd.to_pickle(singleText, path.join(path.dirname(__file__), '..', 'np_data', 'cn_singleText_pd'))

print(text['text_seg'])
print(singleText['singleText_seg'])




