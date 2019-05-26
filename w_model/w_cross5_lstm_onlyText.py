import numpy as np
import tensorflow as tf
from os import path

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code.my_modules import auc, conf_matrix, k_cross_w, get_w_acc_conf,print_conf_matrix,print_dic

w2v_dim = 100
max_count = 30
label_index = 0
train_chunk_number = 5

label = np.load(path.join(path.dirname(__file__), '..', 'w_np_data', 'vectors_label_textvec.npy'))
label = label[:,label_index]
label = np.reshape(label,(len(label),1))
print(label.shape)

dataset = np.load(path.join(path.dirname(__file__), '..', 'w_np_data', 'vectors_wordmatrix.npy'))
print(dataset.shape)




def get_categories(label_index):
    if label_index == 0:
        return 2
    elif label_index == 1:
        return 3


label_categories = get_categories(label_index)

# k_cross trainning

score_list_5chunk = []
confusion_matrix_5chunk = []

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')

for train_chunk_number in range(5):
    train_chunk_number = train_chunk_number + 1

    X_train, X_test = k_cross_w(dataset, train_chunk_number)
    Y_train, Y_test = k_cross_w(label, train_chunk_number)

    # X_train = train[:, 3: (w2v_dim + 3)]
    # X_test = test[:, 3: (w2v_dim + 3)]
    # Y_train = train[:, 0]
    # Y_test = test[:, 0]

    encoder = LabelEncoder()
    encoder_label_train = encoder.fit_transform(Y_train)
    dummy_Y_train = np_utils.to_categorical(encoder_label_train)
    encoder_label_test = encoder.fit_transform(Y_test)
    dummy_Y_test = np_utils.to_categorical(encoder_label_test)


    # 二分类

    def train_text():
        input = Input(shape=(max_count, w2v_dim))
        x = LSTM(128)(input)
        # x = Dropout(0.2)(x)

        y = Dense(label_categories, activation='softmax', name='text_out')(x)
        model = Model(inputs=input, outputs=y)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

        model.fit(X_train, dummy_Y_train, shuffle=True, epochs=50, batch_size=64,
                  validation_split=0.2, verbose=2, callbacks=[early_stopping])

        scores = model.evaluate(X_test, dummy_Y_test, batch_size=64)

        print(model.metrics_names[0] + ":" + str(scores[0]) + "  "
              + model.metrics_names[1] + ":" + str(scores[1]) + "  "
              + model.metrics_names[2] + ":" + str(scores[2]) + "  ")
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]}

        predictions = model.predict(X_test)
        confusion_matrix, acc = get_w_acc_conf(train_chunk_number,label_index,dummy_Y_test,predictions)
        print('w_confusion_matrix =>')
        print(print_conf_matrix(confusion_matrix))
        print('w_acc =>',acc)

        print('round ' + str(train_chunk_number) + ' finished')

        return score_dic, confusion_matrix


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

        print(model.metrics_names[0] + ":" + str(scores[0]) + "  "
              + model.metrics_names[1] + ":" + str(scores[1]) + "  "
              + model.metrics_names[2] + ":" + str(scores[2]) + "  ")
        score_dic = {model.metrics_names[0]: scores[0],
                     model.metrics_names[1]: scores[1],
                     model.metrics_names[2]: scores[2]}

        predictions = model.predict(X_test)
        confusion_matrix = conf_matrix(dummy_Y_test, predictions, label_index)
        print(confusion_matrix)
        return score_dic, confusion_matrix


    score_list_1, confusion_matrix_1 = train_text()
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
print('average final_out_acc = ' + str(final_out_acc / 5))
# print('average auxi_out_acc = '+str(auxi_out_acc / 5))
