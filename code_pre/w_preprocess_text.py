# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from os import path
import re
from code_pre.my_modules import load_stop_words,sentence2wordmatrix
import jieba
from gensim.models import KeyedVectors


## 主要是处理出微博粒度的向量出来，这个微博粒度的词向量组成的矩阵文件特别大，不建议在本机运行
## 输入：label_text_vec_count_pd, w2v_onlycn_100_c.bin
## 输出：(全是w_np_data下的)size.npy, label_text_vectors_pd, label_text_vectors_wordmatrix_pd,
      # vectors_label_textvec.npy, vectors_wordmatrix.npy(5G)

word2vec_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                    'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')


data = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_vector_count_pd'))
print(data)

w2v_dim = 100
max_word = 30


##清理一下text_single中向量为0的文本
def del_empty(sentence_list):
    former_count = len(sentence_list)

    effective_sentence_list = []
    for sentence in sentence_list:
        word_list = sentence.split(' ')
        if sentence=='blank':
            print('blank')
            continue
        elif len(word_list)==0:
            continue
        else:
            effective_word_count = 0
            for word in word_list:
                try:
                    vector = word2vec_model[word]
                    effective_word_count = effective_word_count + 1
                except BaseException as e:
                    pass
            if effective_word_count == 0:
                continue
        effective_sentence_list.append(sentence)

    latter_count = len(effective_sentence_list)
    print('former_count = '+str(former_count))
    print('latter_count = '+str(latter_count))
    return effective_sentence_list


def doc2vec_(sentence, model):
    words = sentence.split(' ')
    vec_sum = np.zeros((w2v_dim,))
    effective_count = 0
    for word in words:
        try:
            vec = model[word]
            vec_sum = np.add(vec_sum, vec)
            effective_count = effective_count + 1
        except BaseException as e:
            pass
    if effective_count == 0:
        print('no word lefts??????')
        print(words)
        return np.zeros((w2v_dim))
    else:
        return np.divide(vec_sum, effective_count)


data['text_single'] = data['text_single'].apply(lambda sentence_list: del_empty(sentence_list))
data['count'] = data['text_single'].apply(lambda sentence_list: len(sentence_list))

print('new_data is updated')


## 下面是要得到每句话的向量组成的矩阵

text_list = []
gender_list = []
age_list = []
area_list = []
size_list = list(data['count'])
print(size_list)
for i in range(len(data)):
    gender = data[1][i]#1 is gender
    age = data[2][i]
    area = data[3][i]

    n = len(data['text_single'][i])
    for j in range(n):
        gender_list.append(gender)
        age_list.append(age)
        area_list.append(area)

    if not len(data['text_single'][i])==n:
        print('???')
        exit(1)

    for sentence in data['text_single'][i]:
        if sentence[-1] == '\n':
            sentence = sentence[: -1]  # delete \n
            print('delete return')
        text_list.append(sentence)

print('text_list len: '+str(len(text_list)))
print('gender_list len: '+str(len(gender_list)))
print('age_list len: '+str(len(age_list)))
print('area_list len: '+str(len(area_list)))
print('size_sum:'+str(np.sum(list(size_list))))
sum = 0
for i in size_list:
    sum = sum + i
print('total size: '+str(sum))

new_data = pd.DataFrame(data=gender_list, columns=['gender'])
new_data['age'] = age_list
new_data['area'] = area_list
new_data['text'] = text_list


size = np.array(size_list)
print(size.shape)
np.save(path.join(path.dirname(__file__),'..','w_np_data','size'),size)


new_data['vectors'] = new_data['text'].apply(lambda sentence: doc2vec_(sentence, word2vec_model))
pd.to_pickle(new_data, path.join(path.dirname(__file__),'..','w_np_data','label_text_vectors_pd'))
##因为下面的wordmatrix太大了，所以这里先保存一份没有wordmatrix的

new_data['wordmatrix'] = new_data['text'].apply(lambda sentence:
                                                sentence2wordmatrix(sentence,word2vec_model,w2v_dim,max_word))
pd.to_pickle(new_data, path.join(path.dirname(__file__),'..','w_np_data','label_text_vectors_wordmatrix_pd'))


gender = np.array(list(new_data['gender']))
gender = gender.reshape((len(gender),1))
age = np.array(list(new_data['age']))
age = age.reshape((len(age),1))
area = np.array(list(new_data['area']))
area = area.reshape((len(area),1))
vectors = np.array(list(new_data['vectors']))


vectors = np.concatenate((gender, age, area, vectors), axis=1)
print(vectors)
print(vectors.shape)
np.save(path.join(path.dirname(__file__),'..','w_np_data','vectors_label_textvec'), vectors)


wordmatrix = np.array(list(new_data['wordmatrix']))
print(wordmatrix)
print(wordmatrix.shape)
np.save(path.join(path.dirname(__file__),'..','w_np_data','vectors_wordmatrix'), wordmatrix)


