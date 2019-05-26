import numpy as np
import tensorflow as tf
from os import path
import pandas as pd

from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,LSTM,Conv2D,AveragePooling2D,Flatten
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code_pre.my_modules import auc, conf_matrix, k_cross, print_conf_matrix, print_dic, k_cross_3

## 输入：vectors_label_text_vec


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

data = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_pd'))
# print(data)
word2vec_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                    'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')

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
# data['text_single_clean'] = data['text_single'].apply(lambda sentence_list: clean_sentence_list(sentence_list))
print('已经清理掉没有对应向量的大小文本集合')


def copy_list(list):
    new_list = []
    for i in list:
        new_list.append(i)
    return new_list

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
# data['text_single_clean_vec'] = data['text_single_clean'].apply(lambda sentence_list: doclist2vec_here(sentence_list,word2vec_model))






w2v_dim = 100
label_index = 0
train_chunk_number = 5

maxlen = 100

label = np.load(path.join(path.dirname(__file__), '..','np_data','vectors_label_textvec.npy'))
label = label[:,label_index]
label = np.reshape(label,(len(label),1))
print(label.shape)
print(label[:])


print(data['text_all_clean_vec'])

user_sequence = np.zeros((len(data),maxlen,w2v_dim))

for i in range(len(data['text_all_clean_vec'])):
    list = data['text_all_clean_vec'][i]
    for j in range(maxlen):
        if j >= len(list):
            pass
        else:
            user_sequence[i,j,:] = np.array(list[j])

print(user_sequence.shape)


def get_categories(label_index):
    if label_index == 0:
        return 2
    elif label_index == 1:
        return 3
label_categories = get_categories(label_index)



# k_cross trainning

f = open(path.join(path.dirname(__file__), '..','record','temp.txt'), 'w')
# for n_epochs in [20,30,40,50,60,70,75,80,90,100,150,200,250,300]:
for n_epochs in [15,20,30]:
    score_list_5chunk = []
    confusion_matrix_5chunk = []

    early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=0, mode='min')

    for train_chunk_number in range(5):
        train_chunk_number = train_chunk_number + 1

        Y_train, Y_test = k_cross(label, train_chunk_number)
        X_train, X_test = k_cross_3(user_sequence, train_chunk_number)
        print(X_train.shape)
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

        encoder = LabelEncoder()
        encoder_label_train = encoder.fit_transform(Y_train)
        dummy_Y_train = np_utils.to_categorical(encoder_label_train)
        encoder_label_test = encoder.fit_transform(Y_test)
        dummy_Y_test = np_utils.to_categorical(encoder_label_test)


        # 二分类

        def train_text_lstm():
            inputs = Input(shape=(maxlen, w2v_dim))
            x = LSTM(128, name='lstm_out')(inputs)
            x = Dropout(0.4)(x)
            x = Dense(50, activation='relu', name='dense_50')(x)
            y = Dense(label_categories, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=y)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

            model.fit(X_train, dummy_Y_train, shuffle=True, epochs=n_epochs, batch_size=64,
                      validation_split=0.2, verbose=2, callbacks=[early_stopping])

            scores = model.evaluate(X_test, dummy_Y_test, batch_size=64)

            print(model.metrics_names[0] + ":" + str(scores[0]) + "  "
                  + model.metrics_names[1] + ":" + str(scores[1]) + "  "
                  + model.metrics_names[2] + ":" + str(scores[2]) + "  ")
            score_dic = {model.metrics_names[0]: scores[0],
                         model.metrics_names[1]: scores[1],
                         model.metrics_names[2]: scores[2]}

            predictions = model.predict(X_test)
            confusion_matrix = conf_matrix(dummy_Y_test, predictions, label_index)
            print(confusion_matrix)
            print('round '+str(train_chunk_number)+' finished')
            return score_dic,confusion_matrix


        def train_text_cnn():
            inputs = Input(shape=(maxlen, w2v_dim,1))
            x = Conv2D(filters=32, kernel_size=(5, 100), activation='relu', name='conv_out')(inputs)
            x = AveragePooling2D((4, 1), name='pool_out')(x)
            x = Flatten()(x)
            x = Dense(50, activation='relu', name='dense_50')(x)
            y = Dense(label_categories, activation='sigmoid', name='final_out')(x)

            model = Model(inputs=inputs, outputs=y)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

            model.fit(X_train, dummy_Y_train, shuffle=True, epochs=n_epochs, batch_size=64,
                      validation_split=0.2, verbose=2, callbacks=[early_stopping])

            scores = model.evaluate(X_test, dummy_Y_test, batch_size=64)

            print(model.metrics_names[0] + ":" + str(scores[0]) + "  "
                  + model.metrics_names[1] + ":" + str(scores[1]) + "  "
                  + model.metrics_names[2] + ":" + str(scores[2]) + "  ")
            score_dic = {model.metrics_names[0]: scores[0],
                         model.metrics_names[1]: scores[1],
                         model.metrics_names[2]: scores[2]}

            predictions = model.predict(X_test)
            confusion_matrix = conf_matrix(dummy_Y_test, predictions, label_index)
            print(confusion_matrix)
            print('round '+str(train_chunk_number)+' finished')
            return score_dic,confusion_matrix




        # 三分类，需要调一下
        def triple_classification():
            model = Sequential()
            model.add(Dense(100, input_dim=w2v_dim, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(50, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(10, activation='relu'))
            # model.add(Dropout(0.4))
            model.add(Dense(label_categories, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])


            model.fit(X_train, dummy_Y_train, shuffle=True, epochs=n_epochs, batch_size=64,
                      validation_split=0.2, verbose=2, callbacks=[early_stopping])

            scores = model.evaluate(X_test, dummy_Y_test, batch_size=64)

            print(model.metrics_names[0]+":"+str(scores[0])+"  "
                  +model.metrics_names[1]+":"+str(scores[1])+"  "
                  +model.metrics_names[2]+":"+str(scores[2])+"  ")
            score_dic = {model.metrics_names[0]: scores[0],
                         model.metrics_names[1]: scores[1],
                         model.metrics_names[2]: scores[2]}

            predictions = model.predict(X_test)
            confusion_matrix = conf_matrix(dummy_Y_test, predictions, label_index)
            print(confusion_matrix)
            return score_dic, confusion_matrix


        score_list_1, confusion_matrix_1 = train_text_cnn()
        score_list_5chunk.append(score_list_1)
        confusion_matrix_5chunk.append(confusion_matrix_1)
        # play_sound()

    print('n_epochs = '+str(n_epochs))
    f.write('n_epochs = '+str(n_epochs)+'----------------------------\n')

    final_out_acc = 0
    # auxi_out_acc = 0
    for i in range(5):
        print(score_list_5chunk[i])
        f.write(print_dic(score_list_5chunk[i]))
        f.write('\n')
        print(confusion_matrix_5chunk[i])
        f.write(print_conf_matrix(confusion_matrix_5chunk[i]))
        f.write('\n')
        final_out_acc = final_out_acc + score_list_5chunk[i]['acc']
        # auxi_out_acc = auxi_out_acc + score_list_5chunk[i]['acc']
    print('average final_out_acc = '+str(final_out_acc / 5))
    f.write('average final_out_acc = '+str(final_out_acc / 5))
    f.write('\n')
    # print('average auxi_out_acc = '+str(auxi_out_acc / 5))
    # f.write('average auxi_out_acc = '+str(auxi_out_acc / 5))
    # f.write('\n')
    f.write('-------------------------------------------\n')
f.close()