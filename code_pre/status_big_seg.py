# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from os import path
import re
from code_pre.my_modules import load_stop_words
import jieba


with open(path.join(path.dirname(__file__),'..','data','status_big.txt'),'r') as f:
    with open(path.join(path.dirname(__file__),'..','data','status_big_seg.txt'),'w') as fw:
        count = 0
        for sentence in f:
            count = count + 1
            cn_char_list = re.findall(r'[\u4e00-\u9fa5\s]', sentence)  # become a char list
            sentence = ''.join(cn_char_list)

            move = dict.fromkeys((ord(c) for c in u'\xa0'))  # delete \xa0
            sentence = sentence.translate(move)

            seg_list = jieba.cut(sentence, cut_all=False)
            stop_words = load_stop_words()
            remain_list = []
            for word in seg_list:  # 清除停用词
                if word == '':
                    pass
                elif word == ' ':
                    pass
                elif word in stop_words:
                    pass
                else:
                    remain_list.append(word)
                outstr = ' '.join(remain_list)
            if count%1000 == 0:
                print(count,outstr)
            fw.write(outstr)


