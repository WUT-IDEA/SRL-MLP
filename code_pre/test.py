# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from os import path
import re
from code.my_modules import load_stop_words
import jieba


with open(path.join(path.dirname(__file__),'..','data','status_big.txt'),'r') as f:
    with open(path.join(path.dirname(__file__),'..','data','status_big_seg.txt'),'w') as fw:
        count = 0
        for sentence in f:
            count = count + 1
            outstr = ''
            sentence = re.findall(r'[\u4e00-\u9fa5\s]', sentence)  # become a char list
            sentence = ''.join(sentence)
            seg_list = jieba.cut(sentence, cut_all=False)
            stop_words = load_stop_words()
            for word in seg_list:  # 清除停用词
                if word not in stop_words:
                    if not word == '\t':
                        outstr = outstr + word.strip()
                        outstr = outstr + " "
            if count%1000 == 0:
                print(str(count)+" ==> "+outstr)
            fw.write(outstr)


