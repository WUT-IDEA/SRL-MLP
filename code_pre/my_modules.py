# -*- coding:utf-8 -*-

import re
import tensorflow as tf
import numpy as np
import pandas as pd
import keras as K
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from os import path
import jieba
import math
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler

def load_stop_words():
    stop_words = list()
    with open(path.join(path.dirname(__file__),'..','data','stopWord.txt'), "r", encoding='utf-8') as stop_f:
        for line in stop_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            stop_words.append(line)
    return stop_words


def extract_cn(sentence):
    if sentence[-1]=='\n':
        sentence = sentence[:-1]
    cn_char_list = re.findall(r'[\u4e00-\u9fa5\s]', sentence)# become a char list
    sentence = ''.join(cn_char_list)

    move = dict.fromkeys((ord(c) for c in u'\xa0'))#delete \xa0
    sentence = sentence.translate(move)


    word_list = sentence.split(' ')
    stop_words = load_stop_words()
    remain_list = []
    for word in word_list:
        # print(word)
        if word == '':
            pass
        elif word == ' ':
            pass
        elif word in stop_words:
            pass
        else:
            remain_list.append(word)
    s = ' '.join(remain_list)
    return s


def extract_cn_jd(sentence):
    if sentence[-1]=='\n':
        sentence = sentence[:-1]
    cn_char_list = re.findall(r'[\u4e00-\u9fa5\s]', sentence)# become a char list
    sentence = ''.join(cn_char_list)

    move = dict.fromkeys((ord(c) for c in u'\xa0'))#delete \xa0
    sentence = sentence.translate(move)


    word_list = jieba.cut(sentence, cut_all=False)
    stop_words = load_stop_words()
    remain_list = []
    for word in word_list:
        if word == '':
            pass
        elif word == ' ':
            pass
        elif word in stop_words:
            pass
        else:
            remain_list.append(word)
    s = ' '.join(remain_list)
    return s



def load_w2v(path):
    lines_num = 0
    vectors = {}
    iw = []# words list
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
    return vectors

w2v_dim = 100

def doc2vec(sentence, model):
    if sentence == 'blank':
        print('sentence is blank')
        return np.zeros((w2v_dim))
    else:
        words = sentence.split(' ')
        vec_sum = np.zeros((w2v_dim))
        effective_count = 0
        for word in words:
            # print(word)
            try:
                vec = model[word]
                vec_sum = np.add(vec_sum, vec)
                effective_count = effective_count + 1
                # print('get vector')
            except BaseException as e:
                print('something wrong')
                # pass
        if effective_count == 0:
            # print('no word lefts')
            return np.zeros((w2v_dim))
        else:
            return np.divide(vec_sum, effective_count)



def doclist2vec(sentence_list, model):
    vec_list = []
    for sentence in sentence_list:
        doc_vec = doc2vec(sentence, model)
        vec_list.append(doc_vec)
    return vec_list


def sentence2wordmatrix(sentence,word2vec_model, w2v_dim, max_word):
    if sentence == 'blank':
        print('sentence is blank')
        return np.zeros((max_word,w2v_dim))
    else:
        words = sentence.split(' ')
        if len(words) == 0:
            return np.zeros((max_word,w2v_dim))

        word_matrix = np.zeros((max_word,w2v_dim))
        effective_count = 0
        for word in words:
            try:
                vec = word2vec_model[word]
                word_matrix[effective_count, :] = vec
                effective_count = effective_count + 1
                if effective_count >= 30:
                    # print('30 is not enough')
                    break
            except BaseException as e:
                # print('something wrong')
                pass
        if effective_count == 0:
            return np.zeros((max_word,w2v_dim))
        else:
            return word_matrix




def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P




def conf_matrix(Y, Y_predict, label_number):
    if label_number==0:
        confusion_matrix = np.zeros((2,2))
        for i in range(len(Y)):
            y = -1
            if Y[i,0]==1:
                y=0
            elif Y[i,1]==1:
                y=1
            else:
                print(i,Y[i,:])
                print('Y is wrong')
                exit(1)

            max_index = 0
            if Y_predict[i,1]>Y_predict[i,0]:
                max_index = 1
            y_predict = max_index

            if y == 0 and y_predict == 0:
                confusion_matrix[0, 0] = confusion_matrix[0, 0] + 1
            elif y == 0 and y_predict == 1:
                confusion_matrix[0, 1] = confusion_matrix[0, 1] + 1
            elif y == 1 and y_predict == 0:
                confusion_matrix[1, 0] = confusion_matrix[1, 0] + 1
            elif y == 1 and y_predict == 1:
                confusion_matrix[1, 1] = confusion_matrix[1, 1] + 1
            else:
                print('label error')
                exit(1)
        return confusion_matrix

    elif label_number == 1:
        confusion_matrix = np.zeros((3, 3))
        for i in range(len(Y)):
            y = -1
            if Y[i, 0] == 1:
                y = 0
            elif Y[i, 1] == 1:
                y = 1
            elif Y[i, 2] == 1:
                y = 2
            else:
                print('Y is wrong')
                exit(1)

            max_index = 0
            if Y_predict[i, 1] > Y_predict[i, 0]:
                max_index = 1
            if Y_predict[i, 2] > Y_predict[i, max_index]:
                max_index = 2
            y_predict = max_index

            if y == 0 and y_predict == 0:
                confusion_matrix[0, 0] = confusion_matrix[0, 0] + 1
            elif y == 0 and y_predict == 1:
                confusion_matrix[0, 1] = confusion_matrix[0, 1] + 1
            elif y == 0 and y_predict == 2:
                confusion_matrix[0, 2] = confusion_matrix[0, 2] + 1
            elif y == 1 and y_predict == 0:
                confusion_matrix[1, 0] = confusion_matrix[1, 0] + 1
            elif y == 1 and y_predict == 1:
                confusion_matrix[1, 1] = confusion_matrix[1, 1] + 1
            elif y == 1 and y_predict == 2:
                confusion_matrix[1, 2] = confusion_matrix[1, 2] + 1
            elif y == 2 and y_predict == 0:
                confusion_matrix[2, 0] = confusion_matrix[2, 0] + 1
            elif y == 2 and y_predict == 1:
                confusion_matrix[2, 1] = confusion_matrix[2, 1] + 1
            elif y == 2 and y_predict == 2:
                confusion_matrix[2, 2] = confusion_matrix[2, 2] + 1
            else:
                print('y = '+str(y)+'  y_predict = '+str(y_predict))
                print('label error_2')
                exit(1)
        return confusion_matrix



def k_cross(dataset, train_chunk):
    size = len(dataset)
    chunk_size = int(size/5)
    chunk_1 = dataset[:chunk_size, :]
    chunk_2 = dataset[chunk_size:chunk_size * 2, :]
    chunk_3 = dataset[chunk_size * 2:chunk_size * 3, :]
    chunk_4 = dataset[chunk_size * 3:chunk_size * 4, :]
    chunk_5 = dataset[chunk_size * 4:, :]

    if train_chunk == 1:
        train = np.concatenate((chunk_2, chunk_3, chunk_4, chunk_5), axis=0)
        test = chunk_1
    elif train_chunk == 2:
        train = np.concatenate((chunk_1, chunk_3, chunk_4, chunk_5), axis=0)
        test = chunk_2
    elif train_chunk == 3:
        train = np.concatenate((chunk_1, chunk_2, chunk_4, chunk_5), axis=0)
        test = chunk_3
    elif train_chunk == 4:
        train = np.concatenate((chunk_1, chunk_2, chunk_3, chunk_5), axis=0)
        test = chunk_4
    elif train_chunk == 5:
        train = np.concatenate((chunk_1, chunk_2, chunk_3, chunk_4), axis=0)
        test = chunk_5
    return train, test


def k_cross_3(dataset, train_chunk):
    size = len(dataset)
    chunk_size = int(size/5)
    chunk_1 = dataset[:chunk_size, :, :]
    chunk_2 = dataset[chunk_size:chunk_size * 2, :, :]
    chunk_3 = dataset[chunk_size * 2:chunk_size * 3, :, :]
    chunk_4 = dataset[chunk_size * 3:chunk_size * 4, :, :]
    chunk_5 = dataset[chunk_size * 4:, :, :]

    if train_chunk == 1:
        train = np.concatenate((chunk_2, chunk_3, chunk_4, chunk_5), axis=0)
        test = chunk_1
    elif train_chunk == 2:
        train = np.concatenate((chunk_1, chunk_3, chunk_4, chunk_5), axis=0)
        test = chunk_2
    elif train_chunk == 3:
        train = np.concatenate((chunk_1, chunk_2, chunk_4, chunk_5), axis=0)
        test = chunk_3
    elif train_chunk == 4:
        train = np.concatenate((chunk_1, chunk_2, chunk_3, chunk_5), axis=0)
        test = chunk_4
    elif train_chunk == 5:
        train = np.concatenate((chunk_1, chunk_2, chunk_3, chunk_4), axis=0)
        test = chunk_5
    return train, test



def k_cross_w(dataset, train_chunk):
    size_np = np.load(path.join(path.dirname(__file__),'..','w_np_data','size.npy'))
    n_1 = int(3138/5)
    n_2 = int(3138/5)*2
    n_3 = int(3138/5)*3
    n_4 = int(3138/5)*4

    index_1 = int(np.sum(size_np[: n_1]))
    index_2 = int(np.sum(size_np[: n_2]))
    index_3 = int(np.sum(size_np[: n_3]))
    index_4 = int(np.sum(size_np[: n_4]))
    # index_5 = np.sum(size_np)

    # size = len(dataset)
    chunk_1 = dataset[:index_1, :]
    chunk_2 = dataset[index_1:index_2, :]
    chunk_3 = dataset[index_2:index_3, :]
    chunk_4 = dataset[index_3:index_4, :]
    chunk_5 = dataset[index_4:, :]

    if train_chunk == 1:
        train = np.concatenate((chunk_2, chunk_3, chunk_4, chunk_5), axis=0)
        test = chunk_1
    elif train_chunk == 2:
        train = np.concatenate((chunk_1, chunk_3, chunk_4, chunk_5), axis=0)
        test = chunk_2
    elif train_chunk == 3:
        train = np.concatenate((chunk_1, chunk_2, chunk_4, chunk_5), axis=0)
        test = chunk_3
    elif train_chunk == 4:
        train = np.concatenate((chunk_1, chunk_2, chunk_3, chunk_5), axis=0)
        test = chunk_4
    elif train_chunk == 5:
        train = np.concatenate((chunk_1, chunk_2, chunk_3, chunk_4), axis=0)
        test = chunk_5
    return train, test




def print_dic(dic):
    s=''
    for i in dic:
        s = s+i+': '+str(dic[i])+", "
    return s

def print_conf_matrix(conf_mat):
    s = ''
    if conf_mat.shape==(2,2):
        s = s + str(conf_mat[0, 0]) + " " + str(conf_mat[0, 1]) +'\n'
        s = s + str(conf_mat[1, 0]) + " " + str(conf_mat[1, 1])
    elif conf_mat.shape==(3,3):
        s = s + str(conf_mat[0, 0]) + " " + str(conf_mat[0, 1]) + " " + str(conf_mat[0, 2]) + '\n'
        s = s + str(conf_mat[1, 0]) + " " + str(conf_mat[1, 1]) + " " + str(conf_mat[1, 2]) + '\n'
        s = s + str(conf_mat[2, 0]) + " " + str(conf_mat[2, 1]) + " " + str(conf_mat[2, 2])
    return s



def get_acc(mat):
    sum = 0
    acc = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sum = sum + mat[i,j]
            if i == j:
                acc = acc + mat[i,j]
    return acc/sum


def get_w_acc_conf(chunk_number, label_index, dummy_label, prediction):
    size_np = np.load(path.join(path.dirname(__file__),'..','w_np_data','size.npy'))

    n_1 = int(3138/5)
    n_2 = int(3138/5)*2
    n_3 = int(3138/5)*3
    n_4 = int(3138/5)*4
    n_5 = 3138

    if chunk_number == 1:
        size_slice = size_np[: n_1]
        Y = np.zeros((n_1, label_index + 2))
        Y_prediction = np.zeros((n_1, label_index + 2))

        former_index = 0
        for i in range(len(size_slice)):
            latter_index = former_index + int(size_slice[i])
            prediction_slice = prediction[former_index:latter_index, :]
            label_slice = dummy_label[former_index:latter_index, :]

            y,y_prediction = w_sameuser_combine(label_slice, prediction_slice)
            # print('return:', y,y.shape, y_prediction,y_prediction.shape)
            # y = np.reshape(y,(1,len(y)))
            # y_prediction = np.reshape(y_prediction, (1, len(y_prediction)))
            former_index = latter_index

            # print(y.shape)
            # print(y_prediction.shape)
            Y[i,:] = y
            Y_prediction[i,:] = y_prediction

            # print('all user combine:')
            # print(Y)
            # print(Y_prediction)

        confusion_matrix = conf_matrix(Y,Y_prediction,label_index)
        acc = get_acc(confusion_matrix)

    elif chunk_number == 2:
        size_slice = size_np[n_1:n_2]
        Y = np.zeros((n_2-n_1, label_index + 2))
        Y_prediction = np.zeros((n_2-n_1,label_index+2))

        former_index = 0
        for i in range(len(size_slice)):
            latter_index = former_index + int(size_slice[i])
            prediction_slice = prediction[former_index:latter_index, :]
            label_slice = dummy_label[former_index:latter_index, :]

            y,y_prediction = w_sameuser_combine(label_slice, prediction_slice)
            # print('return:', y, y.shape, y_prediction, y_prediction.shape)
            # y = np.reshape(y, (1, len(y)))
            # y_prediction = np.reshape(y_prediction, (1, len(y_prediction)))
            former_index = latter_index
            Y[i,:] = y
            Y_prediction[i,:] = y_prediction

        confusion_matrix = conf_matrix(Y,Y_prediction,label_index)
        acc = get_acc(confusion_matrix)

    elif chunk_number == 3:
        size_slice = size_np[n_2:n_3]
        Y = np.zeros((n_3-n_2, label_index + 2))
        Y_prediction = np.zeros((n_3-n_2,label_index+2))

        former_index = 0
        for i in range(len(size_slice)):
            latter_index = former_index + int(size_slice[i])
            prediction_slice = prediction[former_index:latter_index, :]
            label_slice = dummy_label[former_index:latter_index, :]

            y, y_prediction = w_sameuser_combine(label_slice, prediction_slice)
            # print('return:', y, y.shape, y_prediction, y_prediction.shape)
            # y = np.reshape(y, (1, len(y)))
            # y_prediction = np.reshape(y_prediction, (1, len(y_prediction)))
            former_index = latter_index
            Y[i,:] = y
            Y_prediction[i,:] = y_prediction

        confusion_matrix = conf_matrix(Y, Y_prediction, label_index)
        acc = get_acc(confusion_matrix)
    elif chunk_number == 4:
        size_slice = size_np[n_3:n_4]
        Y = np.zeros((n_4 - n_3, label_index + 2))
        Y_prediction = np.zeros((n_4 - n_3, label_index + 2))

        former_index = 0
        for i in range(len(size_slice)):
            latter_index = former_index + int(size_slice[i])
            prediction_slice = prediction[former_index:latter_index, :]
            label_slice = dummy_label[former_index:latter_index, :]

            y, y_prediction = w_sameuser_combine(label_slice, prediction_slice)
            # print('return:', y, y.shape, y_prediction, y_prediction.shape)
            # y = np.reshape(y, (1, len(y)))
            # y_prediction = np.reshape(y_prediction, (1, len(y_prediction)))
            former_index = latter_index
            Y[i,:] = y
            Y_prediction[i,:] = y_prediction

        confusion_matrix = conf_matrix(Y, Y_prediction, label_index)
        acc = get_acc(confusion_matrix)
    elif chunk_number == 5:
        size_slice = size_np[n_4:]
        Y = np.zeros((n_5 - n_4, label_index + 2))
        Y_prediction = np.zeros((n_5 - n_4, label_index + 2))

        former_index = 0
        for i in range(len(size_slice)):
            latter_index = former_index + int(size_slice[i])
            prediction_slice = prediction[former_index:latter_index, :]
            label_slice = dummy_label[former_index:latter_index, :]

            y, y_prediction = w_sameuser_combine(label_slice, prediction_slice)
            # print('return:', y, y.shape, y_prediction, y_prediction.shape)
            # y = np.reshape(y, (1, len(y)))
            # y_prediction = np.reshape(y_prediction, (1, len(y_prediction)))
            former_index = latter_index
            Y[i,:] = y
            Y_prediction[i,:] = y_prediction

            # print(Y)
            # print(Y_prediction)

        confusion_matrix = conf_matrix(Y, Y_prediction, label_index)
        acc = get_acc(confusion_matrix)


    return confusion_matrix, acc



def w_sameuser_combine(label_slice, prediction_slice):
    for i in range(len(label_slice)-1):
        if not (label_slice[i,:] == label_slice[i+1,:]).all():
            # print(label_slice)
            print('labels are not same!!')
            break
    # print('label wrong?')
    # print(label_slice[:,:])
    # y = label_slice[0,:]
    # print('fuzhihou,y=')
    # print(y)
    y_prediction = np.sum(prediction_slice,axis=0)
    y_prediction = np.divide(y_prediction, prediction_slice.shape[0])

    return label_slice[0,:],y_prediction


def decom_pca(data,dim):
    # scaler = StandardScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)

    model = pca.PCA(n_components=dim).fit(data)
    data_trans = model.transform(data)

    print(data_trans.shape)
    return data_trans


def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))