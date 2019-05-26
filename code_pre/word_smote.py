import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from os import path
from heapq import nlargest
import sys
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code_pre.my_modules import auc, conf_matrix, k_cross, print_dic, print_conf_matrix, decom_pca, Cosine
from gensim.models import KeyedVectors

k = 5
w2v_dim = 100

data = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_pd'))
# print(data)
word2vec_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                    'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')

new_data = pd.DataFrame(data = data[0])
new_data[0] = data[1]
new_data[1] = 999
new_data[2] = 999


old_male_text_all_clean = []#存放原始的已经清理掉没有对应向量的词语的大文本
old_male_text_single_clean = []#存放原始的已经清理掉没有对应向量的词语的小文本list
old_female_text_all_clean = []
old_female_text_single_clean = []

def clean_sentence(word_list):
    new_list = []
    for word in word_list:
        if word == '':
            pass
        elif word == ' ':
            pass
        else:
            try:
                vec = word2vec_model[word]
                new_list.append(word)
            except BaseException as e:
                # print('not get vector')
                pass
    return new_list

def clean_sentence_list(sentence_list):
    new_sentence_list = []
    for sentence in sentence_list:
        cleaned_sentence = clean_sentence(sentence)
        if not len(cleaned_sentence) == 0:
            new_sentence_list.append(cleaned_sentence)
        else:
            # print('cleaned_sentence len = 0')
            pass
    return new_sentence_list

data['text_all_clean'] = data['text_all'].apply(lambda sentence: clean_sentence(sentence.split(' ')))
data['text_single_clean'] = data['text_single'].apply(lambda sentence_list: clean_sentence_list(sentence_list))
print('已经清理掉没有对应向量的大小文本集合')

vocab = []#存放每个女性的词汇set
for i in range(len(data)):

    if data[1][i] == 1:
        old_male_text_all_clean.append(data['text_all_clean'][i])
        old_male_text_single_clean.append(data['text_single_clean'][i])
    elif data[1][i] == 0:
        temp_set = set()
        for word in data['text_all_clean'][i]:
            temp_set.add(word)
        vocab.append(temp_set)
        old_female_text_all_clean.append(data['text_all_clean'][i])
        old_female_text_single_clean.append(data['text_single_clean'][i])
    else:
        print('error')
        exit(1)


vocab_vec_list = []
#存放上面的vocab里面每个女性对应的set中的每个词对应的vector，并形成list，最后形成每个女性词汇向量list的list
for vocab_set in vocab:
    vocab_vec = []
    for word in vocab_set:
        vocab_vec.append(word2vec_model[word])
    vocab_vec_list.append(vocab_vec)
# print(vocab_vec_list)
print(len(vocab_vec_list))
print(len(vocab_vec_list[0]))

def copy_list(list):
    new_list = []
    for i in list:
        new_list.append(i)
    return new_list

def word_smote_all(corpus):#这里我会把整个女性的word_list送进来，需对要每个女性进行smote，然后形成新的女性集合
    new_corpus_1 = []
    new_corpus_2 = []
    new_corpus_3 = []
    for vocab_i in range(len(corpus)):
        word_list = corpus[vocab_i]  # word_list是某个女性的词list
        new_vec_list_1 = []  # 我要找到这个女性前100个词的新词
        new_vec_list_2 = []  # 我要找到这个女性前100个词的新词
        new_vec_list_3 = []  # 我要找到这个女性前100个词的新词

        if len(word_list) < 100:
            for i in range(len(word_list)):  # maxlen
                word = word_list[i]
                temp_vocab_vec = copy_list(vocab_vec_list[vocab_i])

                sim_list = []  # len跟vocab_vec一样长
                vec = word2vec_model[word]
                for vv in temp_vocab_vec:
                    sim_list.append(Cosine(vec, vv))

                sum_1 = np.zeros((w2v_dim,))
                sum_2 = np.zeros((w2v_dim,))
                sum_3 = np.zeros((w2v_dim,))
                for j in range(k):
                    index = sim_list.index(max(sim_list))
                    vv = temp_vocab_vec[index]
                    sum_1 = np.add(sum_1, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                    sum_2 = np.add(sum_2, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                    sum_3 = np.add(sum_3, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                    del(temp_vocab_vec[index])
                    del(sim_list[index])
                new_vec_list_1.append(np.divide(sum_1, k))
                new_vec_list_2.append(np.divide(sum_2, k))
                new_vec_list_3.append(np.divide(sum_3, k))
        else:
            for i in range(100):#maxlen
                word = word_list[i]
                temp_vocab_vec = copy_list(vocab_vec_list[vocab_i])

                sim_list = []  # temp_vocab_vec
                vec = word2vec_model[word]
                for vv in temp_vocab_vec:
                    sim_list.append(Cosine(vec, vv))

                sum_1 = np.zeros((w2v_dim,))
                sum_2 = np.zeros((w2v_dim,))
                sum_3 = np.zeros((w2v_dim,))
                if len(sim_list)==0:
                    print('sim_list len = 0 ???')
                for j in range(k):
                    index = sim_list.index(max(sim_list))
                    vv = temp_vocab_vec[index]
                    sum_1 = np.add(sum_1, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                    sum_2 = np.add(sum_2, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                    sum_3 = np.add(sum_3, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                    del(temp_vocab_vec[index])
                    del(sim_list[index])
                new_vec_list_1.append(np.divide(sum_1, k))
                new_vec_list_2.append(np.divide(sum_2, k))
                new_vec_list_3.append(np.divide(sum_3, k))
        new_corpus_1.append(new_vec_list_1)
        new_corpus_2.append(new_vec_list_2)
        new_corpus_3.append(new_vec_list_3)
    return new_corpus_1, new_corpus_2, new_corpus_3

def word_smote_single(word_list_list_list):#传进来的是词list的list的list
    new_word_list_list_list_1 = []#我要得到新词集合的集合
    new_word_list_list_list_2 = []  # 我要得到新词集合的集合
    new_word_list_list_list_3 = []  # 我要得到新词集合的集合
    for word_list_list_i in range(len(word_list_list_list)):
        word_list_list = word_list_list_list[word_list_list_i]
        new_word_list_list_1 = []
        new_word_list_list_2 = []
        new_word_list_list_3 = []
        # if len(temp_vocab_vec)<5:
        #     for word_list in word_list_list:
        #         new_word_list = []
        #         for word in word_list:
        #             new_word_list.append(word2vec_model[word])
        #         new_word_list_list.append(new_word_list)
        # else:
        for word_list_i in range(len(word_list_list)):
            word_list = word_list_list[word_list_i]  # word_list是某个女性的某个词集合
            new_vec_list_1 = []  # 我要找到这个女性前100个词的新词
            new_vec_list_2 = []  # 我要找到这个女性前100个词的新词
            new_vec_list_3 = []  # 我要找到这个女性前100个词的新词

            if len(word_list) < 100:
                for i in range(len(word_list)):  # maxlen
                    word = word_list[i]
                    temp_vocab_vec = copy_list(vocab_vec_list[word_list_list_i])
                    sim_list = []  # len跟vocab_vec一样长
                    vec = word2vec_model[word]
                    for vv in temp_vocab_vec:
                        sim_list.append(Cosine(vec, vv))

                    sum_1 = np.zeros((w2v_dim,))
                    sum_2 = np.zeros((w2v_dim,))
                    sum_3 = np.zeros((w2v_dim,))
                    if len(sim_list) == 0:
                        print('sim_list len = 0 ???')
                    for j in range(k):
                        index = sim_list.index(max(sim_list))
                        vv = temp_vocab_vec[index]
                        sum_1 = np.add(sum_1, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                        sum_2 = np.add(sum_2, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                        sum_3 = np.add(sum_3, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                        del (temp_vocab_vec[index])
                        del (sim_list[index])
                    new_vec_list_1.append(np.divide(sum_1, k))
                    new_vec_list_2.append(np.divide(sum_2, k))
                    new_vec_list_3.append(np.divide(sum_3, k))
            else:
                for i in range(100):  # maxlen
                    word = word_list[i]
                    temp_vocab_vec = copy_list(vocab_vec_list[word_list_list_i])
                    sim_list = []  # len跟vocab_vec一样长
                    vec = word2vec_model[word]
                    for vv in temp_vocab_vec:
                        sim_list.append(Cosine(vec, vv))

                    sum_1 = np.zeros((w2v_dim,))
                    sum_2 = np.zeros((w2v_dim,))
                    sum_3 = np.zeros((w2v_dim,))
                    if len(sim_list) == 0:
                        print('sim_list len = 0 ???')
                    for j in range(k):
                        index = sim_list.index(max(sim_list))
                        vv = temp_vocab_vec[index]
                        sum_1 = np.add(sum_1, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                        sum_2 = np.add(sum_2, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                        sum_3 = np.add(sum_3, np.add(vec, np.multiply(np.subtract(vv, vec), np.random.random())))
                        del (temp_vocab_vec[index])
                        del (sim_list[index])
                    new_vec_list_1.append(np.divide(sum_1, k))
                    new_vec_list_2.append(np.divide(sum_2, k))
                    new_vec_list_3.append(np.divide(sum_3, k))
            new_word_list_list_1.append(new_vec_list_1)
            new_word_list_list_2.append(new_vec_list_2)
            new_word_list_list_3.append(new_vec_list_3)
        new_word_list_list_list_1.append(new_word_list_list_1)
        new_word_list_list_list_2.append(new_word_list_list_2)
        new_word_list_list_list_3.append(new_word_list_list_3)
    return new_word_list_list_list_1,new_word_list_list_list_2,new_word_list_list_list_2


def doc2vec_here(words, model):
    if len(words) == 0:
        print('sentence is blank')
        exit(1)
    else:
        vec_list = []
        for word in words:
            vec_list.append(model[word])
        return vec_list

def doclist2vec_here(word_list_list, model):
    vec_list_list = []
    for word_list in word_list_list:
        vec_list_list.append(doc2vec_here(word_list,model))
    return vec_list_list


data['text_all_clean_vec'] = data['text_all_clean'].apply(lambda word_list: doc2vec_here(word_list,word2vec_model))
data['text_single_clean_vec'] = data['text_single_clean'].apply(lambda sentence_list: doclist2vec_here(sentence_list,word2vec_model))


new_data['text_all_vec'] = data['text_all_clean_vec']
new_data['text_single_vec'] = data['text_single_clean_vec']
print(new_data)



new1_all,new2_all,new3_all = word_smote_all(old_female_text_all_clean)
print('h')
new1_single,new2_single,new3_single = word_smote_single(old_female_text_single_clean)
print('hh')

new1 = pd.DataFrame(data=new1_all)
new1[0] = 0
new1[1] = 999
new1[2] = 999
new1['text_all_vec'] = new1_all
new1['text_single_vec'] = new1_single
print(new1)
new_data = new_data.append(new1,ignore_index=True)

new2 = pd.DataFrame(data=new2_all)
new2[0] = 0
new2[1] = 999
new2[2] = 999
new2['text_all_vec'] = new2_all
new2['text_single_vec'] = new2_single
print(new2)
new_data = new_data.append(new2,ignore_index=True)

new3 = pd.DataFrame(data=new3_all)
new3[0] = 0
new3[1] = 999
new3[2] = 999
new3['text_all_vec'] = new3_all
new3['text_single_vec'] = new3_single
print(new3)
new_data = new_data.append(new3,ignore_index=True)

print(new_data)
pd.to_pickle(new_data,path.join(path.dirname(__file__),'..','np_data','smote_new_label_vectors_pd'))
