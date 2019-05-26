# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import keras
import sys
from code_pre.my_modules import auc, conf_matrix, k_cross, k_cross_3, print_dic, print_conf_matrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from os import path
import h5py

print(sys.argv[1])
command = sys.argv[1]

batch_size = 512
# train_num = 15000
word2vec_dim = 100
maxlen = 100
label_index = 0
train_chunk_number = 5

def get_categories(label_index):
    if label_index == 0:
        return 2
    elif label_index == 1:
        return 3
label_categories = get_categories(label_index)



dataset = np.load(path.join(path.dirname(__file__), '..', 'np_data', 'smote_user_label_vectors.npy'))
label = dataset[:,label_index]
label = label.reshape((len(label),1))

text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','smote_new_label_vectors_pd'))
dynamic_lstm_dataset = np.zeros((len(text),maxlen,word2vec_dim))
for i in range(len(text)):
    dynamic_lstm_dataset[i,:,:] = text['user_granu_vector_sequence'][i]
print(dynamic_lstm_dataset.shape)


from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Input,Conv2D,AveragePooling2D,Flatten
from keras.layers import LSTM

f = open(path.join(path.dirname(__file__), '..', 'record', 'temp.txt'), 'w')

# for n_epochs in [20,30,40,50,60,70,75,80,90,100,150,200,250,300]:
for n_epochs in [15,20,30]:
    score_list_5chunk = []
    confusion_matrix_5chunk = []

    for train_chunk_number in range(5):
        train_chunk_number = train_chunk_number + 1

        # train_text, test_text = k_cross(dataset, train_chunk_number)
        X_train_lstm, X_test_lstm = k_cross_3(dynamic_lstm_dataset, train_chunk_number)
        # X_train_text = train_text[:, 3: 103]
        # X_test_text = test_text[:, 3: 103]
        # Y_train = train_text[:, label_index]
        # Y_test = test_text[:, label_index]


        train_text, test_text = k_cross(dataset, train_chunk_number)
        Y_train, Y_test = k_cross(label, train_chunk_number)
        X_train_cnn, X_test_cnn = k_cross_3(dynamic_lstm_dataset, train_chunk_number)
        X_train_cnn = X_train_cnn.reshape((X_train_cnn.shape[0],X_train_cnn.shape[1],X_train_cnn.shape[2],1))
        X_test_cnn = X_test_cnn.reshape((X_test_cnn.shape[0],X_test_cnn.shape[1],X_test_cnn.shape[2],1))
        X_train_text = train_text[:, 3: 103]
        X_test_text = test_text[:, 3: 103]



        # print(X_train_lstm.shape)


        encoder = LabelEncoder()
        encoder_label_train = encoder.fit_transform(Y_train)
        dummy_Y_train = np_utils.to_categorical(encoder_label_train)
        encoder_label_test = encoder.fit_transform(Y_test)
        dummy_Y_test = np_utils.to_categorical(encoder_label_test)


        # 二分类--------------------

        # 单纯加入2维的情感极性，分别经过一些层，然后再concatenate
        def train_insertlstm():
            # 加载lstm模型
            lstm_inputs = Input(shape=(maxlen,word2vec_dim))
            lstm_lstm = LSTM(128, name='lstm_out')(lstm_inputs)
            lstm_model = Model(inputs=lstm_inputs, outputs=lstm_lstm)

            lstm_model.load_weights(path.join(path.dirname(__file__), 'smote_onlyJD_sentiment_lstm.h5'), by_name=True)
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



        def train_onlyText_lstm():
            # 加载lstm模型
            inputs = Input(shape=(maxlen,word2vec_dim))
            x = LSTM(128, name='lstm_out')(inputs)
            x = Dropout(0.4)(x)
            x = Dense(50,activation='relu')(x)
            y = Dense(label_categories, activation='sigmoid')(x)

            model = Model(inputs=inputs, outputs=y)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',auc])
            model.summary()

            print('fit')
            model.fit(X_train_lstm, dummy_Y_train,shuffle=True,
                      validation_split=0.2, epochs=n_epochs, batch_size=64, verbose=1)

            print('evaluate')
            scores_train = model.evaluate(X_train_lstm, dummy_Y_train, batch_size=64)
            print(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                  + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                  + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                  + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
                  )
            f.write(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                    + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                    + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                    + model.metrics_names[2] + ": " + str(scores_train[2]) + "  \n")

            scores = model.evaluate(X_test_lstm, dummy_Y_test, batch_size=512)
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

            Y_predict = model.predict(X_test_lstm)
            confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
            print(confusion_matrix)
            print('round ' + str(train_chunk_number) + ' finished')
            return score_dic, confusion_matrix



        def train_onlyText_cnn():
            # 加载lstm模型
            inputs = Input(shape=(maxlen, word2vec_dim,1))
            x = Conv2D(filters=32, kernel_size=(5, 100), activation='relu', name='conv_out')(inputs)
            x = AveragePooling2D((4, 1), name='pool_out')(x)
            x = Flatten()(x)
            x = Dense(50, activation='relu', name='dense_50')(x)
            y = Dense(label_categories, activation='sigmoid')(x)

            model = Model(inputs=inputs, outputs=y)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',auc])
            model.summary()

            print('fit')
            model.fit(X_train_cnn, dummy_Y_train,shuffle=True,
                      validation_split=0.2, epochs=n_epochs, batch_size=64, verbose=1)

            print('evaluate')
            scores_train = model.evaluate(X_train_cnn, dummy_Y_train, batch_size=64)
            print(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                  + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                  + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                  + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
                  )
            f.write(str(n_epochs) + ' epochs, ' + str(train_chunk_number) + ' chunk, train_result => '
                    + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
                    + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
                    + model.metrics_names[2] + ": " + str(scores_train[2]) + "  \n")

            scores = model.evaluate(X_test_cnn, dummy_Y_test, batch_size=512)
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

            Y_predict = model.predict(X_test_cnn)
            confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
            print(confusion_matrix)
            print('round ' + str(train_chunk_number) + ' finished')
            return score_dic, confusion_matrix


        # 三分类，需要调一下


        if command == 'insert_lstm':
            score_list_1, confusion_matrix_1 = train_insertlstm()
        elif command == 'onlyText_lstm':
            score_list_1, confusion_matrix_1 = train_onlyText_lstm()
        elif command == 'onlyText_cnn':
            score_list_1, confusion_matrix_1 = train_onlyText_cnn()
        else:
            print(command, 'error')
            exit(1)


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

