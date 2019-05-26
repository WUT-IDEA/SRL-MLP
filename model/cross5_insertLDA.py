import numpy as np
import keras
import tensorflow as tf
from os import path

from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code.my_modules import auc, conf_matrix, k_cross


dataset = np.load(path.join(path.dirname(__file__), '..', 'np_data','vectors_label_textvec.npy'))
senti = np.load(path.join(path.dirname(__file__), '..', 'np_data','vectors_senti.npy'))
lda = np.load(path.join(path.dirname(__file__), '..', 'np_data','vectors_lda.npy'))
dataset = np.concatenate((dataset,senti,lda), axis= 1)
print(dataset.shape)

label_index = 0
train_chunk_number = 5

def get_categories(label_index):
    if label_index == 0:
        return 2
    elif label_index == 1:
        return 3
label_categories = get_categories(label_index)



# k_cross trainning

score_list_5chunk = []
confusion_matrix_5chunk = []
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

for train_chunk_number in range(5):
    train_chunk_number = train_chunk_number + 1

    train, test = k_cross(dataset, train_chunk_number)

    X_train_text = train[:, 3: 303]
    X_test_text = test[:, 3: 303]
    X_train_senti = train[:, 303: 305]
    X_test_senti = test[:, 303: 305]
    X_train_lstm = train[:, 305: 433]
    X_test_lstm = test[:, 305: 433]
    X_train_dense = train[:, 433: 483]
    X_test_dense = test[:, 433: 483]
    X_train_lda = train[:, 483:503]
    X_test_lda = test[:, 483:503]
    Y_train = train[:, 0]
    Y_test = test[:, 0]

    encoder = LabelEncoder()
    encoder_label_train = encoder.fit_transform(Y_train)
    dummy_Y_train = np_utils.to_categorical(encoder_label_train)
    encoder_label_test = encoder.fit_transform(Y_test)
    dummy_Y_test = np_utils.to_categorical(encoder_label_test)


    # 二分类--------------------

    # 单纯加入2维的情感极性，形成102维，还是原来的最直接的模型
    def train_text_senti_1():
        input_text = Input(shape=(100,))
        input_senti = Input(shape=(2,))

        input_text_senti = keras.layers.concatenate([input_text,input_senti])
        text_out = Dense(55, input_dim=102, activation='relu')(input_text_senti)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(10, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        y = Dense(label_categories, activation='sigmoid',name='text_out')(text_out)



        model = Model(inputs = [input_text, input_senti], outputs = y)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_senti], dummy_Y_train,
                  shuffle=True, validation_split=0.2, epochs=400, batch_size=512)


        scores_train = model.evaluate([X_train_text, X_train_senti], dummy_Y_train, batch_size=512)
        print('train_result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              )
        scores = model.evaluate([X_test_text, X_test_senti], dummy_Y_test, batch_size=512)
        print('test_result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              )

        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]
                     }

        Y_predict = model.predict([X_test_text, X_test_senti])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix



    # 单纯加入2维的情感极性，分别经过一些层，然后再concatenate
    def train_text_senti_2():
        input_text = Input(shape=(100,))
        input_senti = Input(shape=(2,))

        text_out = Dense(50, input_dim=100, activation='relu')(input_text)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(10, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        auxi_out = Dense(label_categories, activation='sigmoid', name='text_out')(text_out)

        senti_out = Dense(10, input_dim=2, activation='sigmoid')(input_senti)
        senti_out = Dropout(0.4)(senti_out)
        # senti_out = Dense(label_categories, activation='sigmoid', name='senti_out')(senti_out)

        x = keras.layers.concatenate([text_out, senti_out])
        y = Dense(label_categories, activation='sigmoid', name='final_out')(x)

        model = Model(inputs=[input_text, input_senti], outputs=[y, auxi_out])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_senti], [dummy_Y_train, dummy_Y_train],
                  shuffle=True, validation_split=0.2, epochs=350, batch_size=256)

        scores_train = model.evaluate([X_train_text, X_train_senti], [dummy_Y_train, dummy_Y_train], batch_size=512)
        print('result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              + model.metrics_names[3] + ": " + str(scores_train[3]) + "  "
              + model.metrics_names[4] + ": " + str(scores_train[4]) + "  "
              + model.metrics_names[5] + ": " + str(scores_train[5]) + "  "
              + model.metrics_names[6] + ": " + str(scores_train[6]) + "  "
              )

        scores = model.evaluate([X_test_text, X_test_senti], [dummy_Y_test, dummy_Y_test], batch_size=512)
        print('result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              + model.metrics_names[3] + ": " + str(scores[3]) + "  "
              + model.metrics_names[4] + ": " + str(scores[4]) + "  "
              + model.metrics_names[5] + ": " + str(scores[5]) + "  "
              + model.metrics_names[6] + ": " + str(scores[6]) + "  "
              )
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2],
                     model.metrics_names[3]: scores[3],
                     model.metrics_names[4]: scores[4],
                     model.metrics_names[5]: scores[5],
                     model.metrics_names[6]: scores[6]
                     }

        Y_predict = model.predict([X_test_text, X_test_senti])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict[0], label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix




# 仅加入dense层训练

    def train_text_lstm_1():
        input_text = Input(shape=(100,))
        input_lstm = Input(shape=(128,))

        input_text_lstm = keras.layers.concatenate([input_text,input_lstm])
        text_out = Dense(100, input_dim=228, activation='relu')(input_text_lstm)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(20, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        y = Dense(label_categories, activation='sigmoid',name='text_out')(text_out)



        model = Model(inputs = [input_text, input_lstm], outputs = y)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_lstm], dummy_Y_train,
                  shuffle=True, validation_split=0.2, epochs=100, batch_size=256)


        scores_train = model.evaluate([X_train_text, X_train_lstm], dummy_Y_train, batch_size=256)
        print('train_result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              )
        scores = model.evaluate([X_test_text, X_test_lstm], dummy_Y_test, batch_size=256)
        print('test_result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              )

        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]
                     }

        Y_predict = model.predict([X_test_text, X_test_lstm])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix



    # 单纯加入2维的情感极性，分别经过一些层，然后再concatenate
    def train_text_lstm_2():
        input_text = Input(shape=(100,))
        input_lstm = Input(shape=(128,))

        text_out = Dense(50, input_dim=100, activation='relu')(input_text)
        text_out = Dropout(0.4)(text_out)
        auxi_out = Dense(10, activation='relu')(text_out)
        # auxi_out = Dropout(0.4)(auxi_out)
        auxi_out = Dense(label_categories, activation='sigmoid', name='auxi_out')(auxi_out)

        lstm_out = Dense(50, input_dim=128, activation='relu')(input_lstm)
        lstm_out = Dropout(0.4)(lstm_out)
        # lstm_out = Dense(10, activation='relu')(lstm_out)
        # lstm_out = Dropout(0.4)(lstm_out)
        # senti_out = Dense(label_categories, activation='sigmoid', name='senti_out')(senti_out)

        x = keras.layers.concatenate([text_out, lstm_out])
        x = Dense(10, activation='relu')(x)
        y = Dense(label_categories, activation='sigmoid', name='final_out')(x)

        model = Model(inputs=[input_text, input_lstm], outputs=[y, auxi_out])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_lstm], [dummy_Y_train, dummy_Y_train],
                  shuffle=True, validation_split=0.2, epochs=75, batch_size=256, verbose=2)

        scores_train = model.evaluate([X_train_text, X_train_lstm], [dummy_Y_train, dummy_Y_train], batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              + model.metrics_names[3] + ": " + str(scores_train[3]) + "  "
              + model.metrics_names[4] + ": " + str(scores_train[4]) + "  "
              + model.metrics_names[5] + ": " + str(scores_train[5]) + "  "
              + model.metrics_names[6] + ": " + str(scores_train[6]) + "  "
              )

        scores = model.evaluate([X_test_text, X_test_lstm], [dummy_Y_test, dummy_Y_test], batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              + model.metrics_names[3] + ": " + str(scores[3]) + "  "
              + model.metrics_names[4] + ": " + str(scores[4]) + "  "
              + model.metrics_names[5] + ": " + str(scores[5]) + "  "
              + model.metrics_names[6] + ": " + str(scores[6]) + "  "
              )
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2],
                     model.metrics_names[3]: scores[3],
                     model.metrics_names[4]: scores[4],
                     model.metrics_names[5]: scores[5],
                     model.metrics_names[6]: scores[6]
                     }

        Y_predict = model.predict([X_test_text, X_test_lstm])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict[0], label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix

    def train_text_lstm_3():
        input_text = Input(shape=(100,))
        input_lstm = Input(shape=(128,))

        text_out = Dense(50, input_dim=100, activation='relu')(input_text)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(10, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        # auxi_out = Dense(label_categories, activation='sigmoid', name='auxi_out')(auxi_out)

        lstm_out = Dense(50, input_dim=128, activation='relu')(input_lstm)
        lstm_out = Dropout(0.4)(lstm_out)
        lstm_out = Dense(10, activation='relu')(lstm_out)
        lstm_out = Dropout(0.4)(lstm_out)
        # senti_out = Dense(label_categories, activation='sigmoid', name='senti_out')(senti_out)

        x = keras.layers.concatenate([text_out, lstm_out])
        # x = Dense(10, activation='relu')(x)
        y = Dense(label_categories, activation='sigmoid')(x)

        model = Model(inputs=[input_text, input_lstm], outputs=y)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_lstm], dummy_Y_train,
                  shuffle=True, validation_split=0.2, epochs=50, batch_size=256, verbose=2)

        scores_train = model.evaluate([X_train_text, X_train_lstm], dummy_Y_train, batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              )

        scores = model.evaluate([X_test_text, X_test_lstm], dummy_Y_test, batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              )
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]
                     }

        Y_predict = model.predict([X_test_text, X_test_lstm])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix





    def train_text_dense_1():
        input_text = Input(shape=(100,))
        input_dense = Input(shape=(50,))

        input_text_dense = keras.layers.concatenate([input_text, input_dense])
        text_out = Dense(75, input_dim=150, activation='relu')(input_text_dense)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(15, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        y = Dense(label_categories, activation='sigmoid',name='text_out')(text_out)



        model = Model(inputs = [input_text, input_dense], outputs = y)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_dense], dummy_Y_train,
                  shuffle=True, validation_split=0.2, epochs=30, batch_size=256)


        scores_train = model.evaluate([X_train_text, X_train_dense], dummy_Y_train, batch_size=256)
        print('train_result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              )
        scores = model.evaluate([X_test_text, X_test_dense], dummy_Y_test, batch_size=256)
        print('test_result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              )

        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]
                     }

        Y_predict = model.predict([X_test_text, X_test_dense])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix



    # 单纯加入2维的情感极性，分别经过一些层，然后再concatenate
    def train_text_dense_2():
        input_text = Input(shape=(100,))
        input_dense = Input(shape=(50,))

        text_out = Dense(50, input_dim=100, activation='relu')(input_text)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(10, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        auxi_out = Dense(label_categories, activation='sigmoid', name='auxi_out')(text_out)

        dense_out = Dense(10, input_dim=50, activation='relu')(input_dense)
        dense_out = Dropout(0.4)(dense_out)
        # lstm_out = Dense(10, activation='relu')(lstm_out)
        # lstm_out = Dropout(0.4)(lstm_out)
        # senti_out = Dense(label_categories, activation='sigmoid', name='senti_out')(senti_out)

        x = keras.layers.concatenate([text_out, dense_out])
        # x = Dense(10, activation='relu')(x)
        y = Dense(label_categories, activation='sigmoid', name='final_out')(x)

        model = Model(inputs=[input_text, input_dense], outputs=[y, auxi_out])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_dense], [dummy_Y_train, dummy_Y_train],
                  shuffle=True, validation_split=0.2, epochs=200, batch_size=256, verbose=2)

        scores_train = model.evaluate([X_train_text, X_train_dense], [dummy_Y_train, dummy_Y_train], batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              + model.metrics_names[3] + ": " + str(scores_train[3]) + "  "
              + model.metrics_names[4] + ": " + str(scores_train[4]) + "  "
              + model.metrics_names[5] + ": " + str(scores_train[5]) + "  "
              + model.metrics_names[6] + ": " + str(scores_train[6]) + "  "
              )

        scores = model.evaluate([X_test_text, X_test_dense], [dummy_Y_test, dummy_Y_test], batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              + model.metrics_names[3] + ": " + str(scores[3]) + "  "
              + model.metrics_names[4] + ": " + str(scores[4]) + "  "
              + model.metrics_names[5] + ": " + str(scores[5]) + "  "
              + model.metrics_names[6] + ": " + str(scores[6]) + "  "
              )
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2],
                     model.metrics_names[3]: scores[3],
                     model.metrics_names[4]: scores[4],
                     model.metrics_names[5]: scores[5],
                     model.metrics_names[6]: scores[6]
                     }

        Y_predict = model.predict([X_test_text, X_test_dense])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict[0], label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix

    def train_text_dense_3():
        input_text = Input(shape=(100,))
        input_dense = Input(shape=(50,))

        text_out = Dense(50, input_dim=100, activation='relu')(input_text)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(10, activation='relu')(text_out)
        # auxi_out = Dropout(0.4)(auxi_out)
        # auxi_out = Dense(label_categories, activation='sigmoid', name='auxi_out')(auxi_out)

        lstm_out = Dense(10, input_dim=50, activation='relu')(input_dense)
        # lstm_out = Dropout(0.4)(lstm_out)
        # lstm_out = Dense(10, activation='relu')(lstm_out)
        # lstm_out = Dropout(0.4)(lstm_out)
        # senti_out = Dense(label_categories, activation='sigmoid', name='senti_out')(senti_out)

        x = keras.layers.concatenate([text_out, lstm_out])
        # x = Dense(10, activation='relu')(x)
        y = Dense(label_categories, activation='sigmoid')(x)

        model = Model(inputs=[input_text, input_dense], outputs=y)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_dense], dummy_Y_train,
                  shuffle=True, validation_split=0.2, epochs=300, batch_size=256, verbose=2)

        scores_train = model.evaluate([X_train_text, X_train_dense], dummy_Y_train, batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              )

        scores = model.evaluate([X_test_text, X_test_dense], dummy_Y_test, batch_size=256)
        print('result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              )
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]
                     }

        Y_predict = model.predict([X_test_text, X_test_dense])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix



    def train_text_lda_1():
        input_text = Input(shape=(300,))
        input_lda = Input(shape=(20,))

        input_text_senti = keras.layers.concatenate([input_text,input_lda])
        text_out = Dense(100, input_dim=302, activation='relu')(input_text_senti)
        text_out = Dropout(0.4)(text_out)
        text_out = Dense(20, activation='relu')(text_out)
        text_out = Dropout(0.4)(text_out)
        y = Dense(label_categories, activation='sigmoid',name='text_out')(text_out)



        model = Model(inputs = [input_text, input_lda], outputs = y)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', auc])

        model.fit([X_train_text, X_train_lda], dummy_Y_train,shuffle=True,
                  validation_split=0.2, epochs=300, batch_size=64, callbacks=[early_stopping])


        scores_train = model.evaluate([X_train_text, X_train_lda], dummy_Y_train, batch_size=512)
        print('train_result => ' + model.metrics_names[0] + ": " + str(scores_train[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores_train[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores_train[2]) + "  "
              )
        scores = model.evaluate([X_test_text, X_test_lda], dummy_Y_test, batch_size=512)
        print('test_result => ' + model.metrics_names[0] + ": " + str(scores[0]) + "  "
              + model.metrics_names[1] + ": " + str(scores[1]) + "  "
              + model.metrics_names[2] + ": " + str(scores[2]) + "  "
              )

        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]
                     }

        Y_predict = model.predict([X_test_text, X_test_lda])
        confusion_matrix = conf_matrix(dummy_Y_test, Y_predict, label_index)
        print(confusion_matrix)
        print('round ' + str(train_chunk_number) + ' finished')
        return score_dic, confusion_matrix







    # 三分类，需要调一下







    score_list_1, confusion_matrix_1 = train_text_lda_1()
    score_list_5chunk.append(score_list_1)
    confusion_matrix_5chunk.append(confusion_matrix_1)



final_out_acc = 0
# auxi_out_acc = 0
for i in range(5):
    print(score_list_5chunk[i])
    print(confusion_matrix_5chunk[i])
    final_out_acc = final_out_acc + score_list_5chunk[i]['acc']
    # auxi_out_acc = auxi_out_acc + score_list_5chunk[i]['acc']
print('average final_out_acc = '+str(final_out_acc / 5))
# print('average auxi_out_acc = '+str(auxi_out_acc / 5))
