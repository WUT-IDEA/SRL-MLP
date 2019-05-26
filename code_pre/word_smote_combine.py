import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from os import path
import sys
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code_pre.my_modules import auc, conf_matrix, k_cross, print_dic, print_conf_matrix, decom_pca, Cosine
from gensim.models import KeyedVectors


data = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','smote_temp_pd'))
data = data[:4672]




maxlen = 100
w2v_dim = 100

# vectors = np.zeros((len(data),w2v_dim))



# def vectors_add(word_vector_list):
#     sum = np.zeros((w2v_dim,))
#     for v in word_vector_list:
#         sum = np.add(sum, v)
#     avg = np.divide(sum,len(word_vector_list))
#     return avg



def to_matrix(word_vector_list):
    word_matrix = np.zeros((maxlen,w2v_dim))
    for i in range(maxlen):
        if i >= len(word_vector_list):
            pass
        else:
            word_matrix[i,:] = np.array(word_vector_list[i])
    return word_matrix

def to_matrx_list(word_vector_list_list):
    list = []
    for word_vector_list in word_vector_list_list:
        list.append(to_matrix(word_vector_list))
    return list


label = np.zeros((len(data)))
label[:] = np.array(data[0])
print(label)

new_data = pd.DataFrame(data=label)
new_data[1] = data[1]
new_data[2] = data[2]


smote_tl = np.load(path.join(path.dirname(__file__),'..','np_data','smote_tl.npy'))
smote_tl = smote_tl[:,3:]
print('smote_tl: ')
print(smote_tl)
list = []
for i in range(smote_tl.shape[0]):
    temp = np.zeros((w2v_dim,))
    temp[:] = smote_tl[i,:]
    list.append(temp)
new_data['user_granu_vector'] = list




# new_data['user_granu_vector'] = data['text_all_vec'].apply(lambda word_vector_list: vectors_add(word_vector_list))

new_data['user_granu_vector_sequence'] = data['text_all_vec'].apply(lambda word_vector_list: to_matrix(word_vector_list))
new_data['microblog_granu_vector_sequence'] = data['text_single_vec'].apply(lambda word_vector_list_list: to_matrx_list(word_vector_list_list))
print(new_data)

new_data = new_data.sample(frac=1).reset_index(drop=True)

gender = np.array(new_data[0]).reshape((len(new_data),1))
age = np.array(new_data[1]).reshape((len(new_data),1))
area = np.array(new_data[2]).reshape((len(new_data),1))




user_granu_vector = np.zeros((len(new_data),w2v_dim))
for i in range(len(new_data)):
    user_granu_vector[i,:] = new_data['user_granu_vector'][i]

print('user_granu_vector shape = ',user_granu_vector.shape)
user_granu_vector = np.concatenate((gender,age,area,user_granu_vector), axis=1)

print(user_granu_vector.shape)

np.save(path.join(path.dirname(__file__),'..','np_data','smote_user_label_vectors.npy'),user_granu_vector)

pd.to_pickle(new_data,path.join(path.dirname(__file__),'..','np_data','smote_new_label_vectors_pd'))
