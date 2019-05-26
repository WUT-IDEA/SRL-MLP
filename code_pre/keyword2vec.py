# coding:utf-8

from os import path
import pandas as pd
import numpy as np
import re
from jieba import analyse

data = pd.read_pickle(path.join(path.dirname(__file__), '..', 'np_data', 'label_text_pd'))
# print(data)

##text_all
keywords_list = []
with open(path.join(path.dirname(__file__), '..', 'data', 'words_female.txt'),'r') as f:
    for k in f.readlines():
        keywords_list.append(k)
with open(path.join(path.dirname(__file__), '..', 'data', 'words_male.txt'),'r') as f:
    for k in f.readlines():
        keywords_list.append(k)

def keywords2vector(sentence, keywords_list):
    words = sentence.split(' ')
    vec_len = len(keywords_list)
    vec = np.zeros((vec_len,))
    # for word in words:
    #     for i in range(vec_len):
    #         if word == keywords_list[i]:
    #             print('hh')
    #             vec[i] += 1
    for i in range(vec_len):
        l = re.findall(keywords_list[i],sentence)
        if not len(l)==0:
            print('hh')
        vec[i] = vec[i] + len(l)
    print(vec)
    return vec

data['keywords2vec'] = data['text_all'].apply(lambda sentence: keywords2vector(sentence, keywords_list))

vectors = np.array(list(data['keywords2vec']))
print(vectors)
print(vectors.shape)

for i in range(vectors.shape[0]):
    for j in range(vectors.shape[1]):
        if not vectors[i][j]==0:
            print(str(vectors[i][j]) + ' ')