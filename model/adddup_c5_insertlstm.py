# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import keras
from code_pre.my_modules import auc, conf_matrix, k_cross, print_dic, print_conf_matrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from os import path
import h5py

batch_size = 512
train_num = 15000
word2vec_dim = 100
maxlen = 100

# 用来调试接过来的lstm模型用的，后面可以删掉
# all_ = pd.read_pickle('../np_data/jd_words_doc2num_pd')
# X = np.array(list(all_['doc2num']))
# Y = np.array(list(all_['label']))
# Y = Y.reshape((-1,1)) #调整标签形状

# 准备微博的lstm的数据
dict_index = pd.read_pickle(path.join(path.dirname(__file__), '..', 'np_data', 'adddup_jd_word2seq_pd'))
n_symbols = len(dict_index) + 1
embedding_weights = np.load(path.join(path.dirname(__file__), '..', 'np_data', 'adddup_jd_embedding_weights.npy'))
word_set = set(dict_index.index)


def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + [''] * max(0, maxlen - len(s))
    # print(abc[s])#输出的是一系列词和词对应的编码序号
    # print(list(abc[s]))#输出的是[3,6128,2168,……]这样的序列，如果词不够100，则后面跟的全是0
    return list(dict_index[s])


text = pd.read_pickle(path.join(path.dirname(__file__), '..', 'np_data', 'adddup_label_text_sentistatics_pd'))
# senti_fea1 = np.array(text['text_all_senti'])
# senti_fea1 = senti_fea1.reshape((-1, 1))
#
# senti_fea2 = np.array(text['single_senti_statics'])

text['text2seq'] = text['text_all'].apply(lambda sentence: doc2num(sentence.split(' '), maxlen))
print(text['text2seq'])
lstm_data = np.array(list(text['text2seq']))
print(lstm_data)
print(lstm_data.shape)

# 准备微博的全连接网络的数据
dataset = np.load(path.join(path.dirname(__file__), '..', 'np_data', 'vectors_label_textvec.npy'))
# senti = np.load('../np_data/vectors_cnnsenti.npy')
# dataset = np.concatenate((dataset,senti), axis= 1)
print(dataset.shape)

label_index = 0
train_chunk_number = 5


def get_categories(label_index):
    if label_index == 0:
        return 2
    elif label_index == 1:
        return 3


label_categories = get_categories(label_index)

# 将lstm和fcnn的数据concatenate，因为后面要k_cross
dataset = np.concatenate((dataset, lstm_data), axis=1)

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Input
from keras.layers import LSTM

# k_cross trainning
f = open(path.join(path.dirname(__file__), '..', 'record', 'temp.txt'), 'w')

for n_epochs in [20,30,40,50,60,70,75,80,90,100,150,200,250,300]:
    score_list_5chunk = []
    confusion_matrix_5chunk = []

    for train_chunk_number in range(5):
        train_chunk_number = train_chunk_number + 1

        train, test = k_cross(dataset, train_chunk_number)

        X_train_text = train[:, 3: 103]
        X_test_text = test[:, 3: 103]
        X_train_lstm = train[:, 103: 203]
        X_test_lstm = test[:, 103: 203]
        Y_train = train[:, label_index]
        Y_test = test[:, label_index]

        encoder = LabelEncoder()
        encoder_label_train = encoder.fit_transform(Y_train)
        dummy_Y_train = np_utils.to_categorical(encoder_label_train)
        encoder_label_test = encoder.fit_transform(Y_test)
        dummy_Y_test = np_utils.to_categorical(encoder_label_test)


        # 二分类--------------------

        # 单纯加入2维的情感极性，分别经过一些层，然后再concatenate
        def train_insertlstm():
            # 加载lstm模型
            lstm_inputs = Input(shape=(maxlen,))
            lstm_embedding = Embedding(output_dim=word2vec_dim,
                                       input_dim=n_symbols,
                                       mask_zero=True,
                                       weights=[embedding_weights],
                                       input_length=maxlen, name='embedding_1')(lstm_inputs)
            lstm_lstm = LSTM(128, name='lstm_out')(lstm_embedding)
            lstm_model = Model(inputs=lstm_inputs, outputs=lstm_lstm)

            lstm_model.load_weights(path.join(path.dirname(__file__), 'adddup_sentiment_lstm.h5'), by_name=True)
            print(lstm_model.get_weights())

            # 主要用来分类的全连接层
            fcnn_inputs = Input(shape=(100,), name='fcnn_input')
            # fcnn_dense_a = Dense(50,activation='relu')(fcnn_inputs)
            # fcnn_dropout_a = Dropout(0.4)(fcnn_dense_a)
            # fcnn_dense_aa = Dense(10,activation='relu')(fcnn_dropout_a)
            # lstm_dense_b = Dense(50,activation='relu')(lstm_lstm)
            # lstm_dropout_b = Dropout(0.4)(lstm_dense_b)
            # lstm_dense_bb = Dense(10,activation='relu')(lstm_dropout_b)
            fcnn_conca = keras.layers.concatenate([fcnn_inputs, lstm_lstm])
            x = Dense(100, activation='relu')(fcnn_conca)
            x = Dropout(0.4)(x)
            x = Dense(20, activation='relu')(x)
            fcnn_out = Dense(label_categories, activation='sigmoid')(x)

            model = Model(inputs=[fcnn_inputs, lstm_inputs], outputs=fcnn_out)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',auc])
            model.summary()

            print('fit')
            model.fit([X_train_text, X_train_lstm], dummy_Y_train,shuffle=True,
                      validation_split=0.2, epochs=n_epochs, batch_size=64, verbose=1)

            print('evaluate')
            scores_train = model.evaluate([X_train_text, X_train_lstm], dummy_Y_train, batch_size=64)
            print(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                  + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                  + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                  + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
                  )
            f.write(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                    + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                    + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                    + model.metrics_names[2] + ": " + str(scores_train[2]) + "  \n")

            scores = model.evaluate([X_test_text, X_test_lstm], dummy_Y_test, batch_size=512)
            print(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, test_result => '
                  + model.metrics_names[0] + ": " + str(scores[0]) + "  "
                  + model.metrics_names[1] + ": " + str(scores[1]) + "  "
                  + model.metrics_names[2] + ": " + str(scores[2]) + "  "
                  )
            f.write(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, test_result => '
                    + model.metrics_names[0] + ": " + str(scores[0]) + "  "
                    + model.metrics_names[1] + ": " + str(scores[1]) + "  "
                    + model.metrics_names[2] + ": " + str(scores[2]) + "  \n")

            score_dic = {model.metrics_names[0]: scores[0],
                         model.metrics_names[1]: scores[1],
                         model.metrics_names[2]: scores[2]
                         }

            Y_predict = model.predict([X_test_text, X_test_lstm])
            confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
            print(confusion_matrix)
            print('round ' + str(train_chunk_number) + ' finished')
            return score_dic, confusion_matrix


        def train_insertdense():
            # 加载lstm模型
            lstm_inputs = Input(shape=(maxlen,))
            lstm_embedding = Embedding(output_dim=word2vec_dim,
                                       input_dim=n_symbols,
                                       mask_zero=True,
                                       weights=[embedding_weights],
                                       input_length=maxlen,
                                       name='embedding_1')(lstm_inputs)
            lstm_lstm = LSTM(128, name='lstm_out')(lstm_embedding)
            lstm_dropout_1 = Dropout(0.5)(lstm_lstm)
            lstm_dense_50 = Dense(50, activation='relu', name='dense_50')(lstm_dropout_1)

            lstm_model = Model(inputs=lstm_inputs, outputs=lstm_dense_50)
            lstm_model.load_weights(path.join(path.dirname(__file__), 'adddup_sentiment_lstm.h5'), by_name=True)

            # 主要用来分类的全连接层
            fcnn_inputs = Input(shape=(100,))
            fcnn_conca = keras.layers.concatenate([fcnn_inputs, lstm_dense_50])
            fcnn_dense_100 = Dense(75, activation='relu')(fcnn_conca)
            fcnn_dropout_1 = Dropout(0.4)(fcnn_dense_100)
            fcnn_dense_20 = Dense(15, activation='relu')(fcnn_dropout_1)
            fcnn_dropout_2 = Dropout(0.4)(fcnn_dense_20)
            fcnn_out = Dense(label_categories, activation='sigmoid')(fcnn_dropout_2)

            model = Model(inputs=[fcnn_inputs, lstm_inputs], outputs=fcnn_out)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',auc])
            model.summary()

            print('fit')
            model.fit([X_train_text, X_train_lstm], dummy_Y_train,
                      shuffle=True, validation_split=0.2, epochs=n_epochs, batch_size=64, verbose=1)

            print('evaluate')
            scores_train = model.evaluate([X_train_text, X_train_lstm], dummy_Y_train, batch_size=64)
            print(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                  + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                  + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                  )
            f.write(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                    + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                    + model.metrics_names[1] + ": " + str(scores_train[1]) + "  \n")

            scores = model.evaluate([X_test_text, X_test_lstm], dummy_Y_test, batch_size=512)
            print(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, test_result => '
                  + model.metrics_names[0] + ": " + str(scores[0]) + "  "
                  + model.metrics_names[1] + ": " + str(scores[1]) + "  "
                  )
            f.write(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, test_result => '
                    + model.metrics_names[0] + ": " + str(scores[0]) + "  "
                    + model.metrics_names[1] + ": " + str(scores[1]) + "  \n")

            score_dic = {model.metrics_names[0]: scores[0],
                         model.metrics_names[1]: scores[1]
                         }

            Y_predict = model.predict([X_test_text, X_test_lstm])
            confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
            print(confusion_matrix)
            print('round ' + str(train_chunk_number) + ' finished')
            return score_dic, confusion_matrix


        # 三分类，需要调一下


        score_list_1, confusion_matrix_1 = train_insertlstm()
        score_list_5chunk.append(score_list_1)
        confusion_matrix_5chunk.append(confusion_matrix_1)
        # play_sound()

    print('n_epochs = ' + str(n_epochs))
    f.write('n_epochs = ' + str(n_epochs) + '----------------------------\n')

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
    print('average final_out_acc = ' + str(final_out_acc / 5))
    f.write('average final_out_acc = ' + str(final_out_acc / 5))
    f.write('\n')
    # print('average auxi_out_acc = '+str(auxi_out_acc / 5))
    # f.write('average auxi_out_acc = '+str(auxi_out_acc / 5))
    # f.write('\n')
    f.write('-------------------------------------------\n')
f.close()

