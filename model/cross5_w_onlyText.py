import numpy as np
import tensorflow as tf
from os import path

from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,LSTM,Conv1D,AveragePooling1D,Conv2D,AveragePooling2D,Flatten
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code.my_modules import auc, conf_matrix, k_cross


w2v_dim = 100
max_count = 100 #每个人最多有100条微博
label_index = 0
train_chunk_number = 5

a = np.load(path.join(path.dirname(__file__), '..','np_data','vectors_label_textvec.npy'))
label = a[:, label_index]
label = np.reshape(label,(len(label), 1))#这里如果不reshape则在k_cross中会提示too many indices for array
print(label.shape)

vectors = np.load(path.join(path.dirname(__file__), '..','np_data','vectors_singleTextvec.npy'))


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

    X_train, X_test = k_cross(vectors, train_chunk_number)
    Y_train, Y_test = k_cross(label, train_chunk_number)
    print(Y_train.shape)
    print(Y_test.shape)

    # X_train = X_train.reshape((len(X_train), max_count, w2v_dim))#lstm
    # X_test = X_test.reshape((len(X_test), max_count, w2v_dim))
    X_train = X_train.reshape((len(X_train), max_count, w2v_dim, 1))#cnn
    X_test = X_test.reshape((len(X_test), max_count, w2v_dim, 1))
    Y_train = Y_train.reshape((len(Y_train)))
    Y_test = Y_test.reshape((len(Y_test)))
    print(Y_train.shape)
    print(Y_test.shape)


    # X_train = train[:, 3: (w2v_dim+3)]
    # X_test = test[:, 3: (w2v_dim+3)]
    # Y_train = train[:, label_index]
    # Y_test = test[:, label_index]

    encoder = LabelEncoder()
    encoder_label_train = encoder.fit_transform(Y_train)
    dummy_Y_train = np_utils.to_categorical(encoder_label_train)
    encoder_label_test = encoder.fit_transform(Y_test)
    dummy_Y_test = np_utils.to_categorical(encoder_label_test)


    # 二分类

    def train_text_lstm():
        input = Input(shape=(max_count, w2v_dim))

        x = LSTM(256)(input)
        x = Dropout(0.4)(x)
        x = Dense(100,activation='tanh')(x)
        x = Dense(20,activation='tanh')(x)
        y = Dense(label_categories, activation='tanh', name='text_out')(x)

        model = Model(inputs=input, outputs=y)

        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', auc])

        model.fit(X_train, dummy_Y_train, shuffle=True, epochs=20, batch_size=64,
                  validation_split=0.2, verbose=2)

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
        input = Input(shape=(max_count, w2v_dim, 1))

        x = Conv2D(filters=16, kernel_size=(5,100), activation='tanh')(input)
        x = AveragePooling2D(pool_size=(4,1))(x)
        x = Flatten()(x)
        x = Dense(100,activation='relu')(x)
        x = Dense(10,activation='relu')(x)
        y = Dense(label_categories, activation='tanh', name='text_out')(x)

        model = Model(inputs=input, outputs=y)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])
        model.summary()

        model.fit(X_train, dummy_Y_train, shuffle=True, epochs=10, batch_size=64,
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


        model.fit(X_train, dummy_Y_train, shuffle=True, epochs=800, batch_size=64,
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


# print(score_list_5chunk)
# print(confusion_matrix_5chunk)


final_out_acc = 0
# auxi_out_acc = 0
for i in range(5):
    print(score_list_5chunk[i])
    print(confusion_matrix_5chunk[i])
    final_out_acc = final_out_acc + score_list_5chunk[i]['acc']
    # auxi_out_acc = auxi_out_acc + score_list_5chunk[i][6][1]
print('average final_out_acc = '+str(final_out_acc / 5))
# print('average auxi_out_acc = '+str(auxi_out_acc / 5))
