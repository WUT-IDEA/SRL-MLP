# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from os import path
from gensim.models import KeyedVectors
from gensim.models import word2vec
from code_pre.my_modules import doc2vec, doclist2vec, load_w2v


# 下面是将大文本和小文本向量化的代码，时间不是很长但是还是避免重复向量化
## 输入：label_text_pd, w2v_onlycn_100_c.bin
## 输出：label_text_vector_count_pd, vectors_label_textvec.npy, vectors_singleTextvec.npy(这个没有label)

data = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_pd'))

word2vec_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                    'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')


def trans(label):
    temp = np.zeros((1,))
    temp[0] = label
    return temp


data['doc2vec'] = data['text_all'].apply(lambda sentence: doc2vec(sentence, word2vec_model))
data['doclist2vec'] = data['text_single'].apply(lambda sentence_list: doclist2vec(sentence_list, word2vec_model))
data['count'] = data['text_single'].apply(lambda sentence_list: trans(len(sentence_list)))

data.to_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_vector_count_pd'))

# 已经向量化过了，保存在了label_text_vector_count_pd中，前面的注释掉了，后面的直接读取就好
# data = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_vector_count_pd'))

print(data)




def get_gender():
    data['gender'] = data[1].apply(lambda label: trans(label))
    labels = np.array(list(data['gender']))
    return labels

def get_age():
    data['age'] = data[2].apply(lambda label: trans(label))
    labels = np.array(list(data['age']))
    return labels

def get_area():
    data['area'] = data[3].apply(lambda label: trans(label))
    labels = np.array(list(data['area']))
    return labels

def get_text_vectors():
    vectors = np.array(list(data['doc2vec']))
    return vectors

def get_count_norm():
    count = data['count']
    max = np.max(count)
    print(max)
    count_norm = np.divide(count, max)
    return count_norm


dataset = np.concatenate((get_gender(),
                          get_age(),
                          get_area(),
                          get_text_vectors(),), axis=1)

np.save(path.join(path.dirname(__file__),'..','np_data','vectors_label_textvec'), dataset)






w2v_dim = 100
def get_singleText_vectors():
    n = len(data['doclist2vec'])
    max_count = 100
    v = np.zeros((n,max_count,w2v_dim))

    for i in range(n):
        vec_list = data['doclist2vec'][i]
        count = len(vec_list)
        vectors = np.zeros((max_count,w2v_dim))
        for i in range(max_count):
            if i < count:
                vectors[i, :] = vec_list[i]
            else:
                vectors[i, :] = np.zeros((w2v_dim,))
        v[i, :, :] = vectors
    return v


singleText_dataset = get_singleText_vectors()
print(singleText_dataset.shape)
# print(singleText_dataset)
np.save(path.join(path.dirname(__file__),'..','np_data','vectors_singleTextvec'), singleText_dataset)

