import numpy as np
import tensorflow as tf
from os import path

from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from code_pre.my_modules import auc, conf_matrix, k_cross, print_conf_matrix, print_dic

## 输入：vectors_label_text_vec


w2v_dim = 100
label_index = 0
train_chunk_number = 5

dataset = np.load(path.join(path.dirname(__file__), '..','np_data','vectors_label_textvec.npy'))
print(dataset.shape)
print(dataset[:,label_index])



def get_categories(label_index):
    if label_index == 0:
        return 2
    elif label_index == 1:
        return 3
label_categories = get_categories(label_index)



# k_cross trainning

f = open(path.join(path.dirname(__file__), '..','record','temp.txt'), 'w')
# for n_epochs in [20,30,40,50,60,70,75,80,90,100,150,200,250,300]:
for n_epochs in [50,60]:
    score_list_5chunk = []
    confusion_matrix_5chunk = []

    early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=0, mode='min')

    for train_chunk_number in range(5):
        train_chunk_number = train_chunk_number + 1

        train, test = k_cross(dataset, train_chunk_number)

        X_train = train[:, 3: (w2v_dim+3)]
        X_test = test[:, 3: (w2v_dim+3)]
        Y_train = train[:, 0]
        Y_test = test[:, 0]

        encoder = LabelEncoder()
        encoder_label_train = encoder.fit_transform(Y_train)
        dummy_Y_train = np_utils.to_categorical(encoder_label_train)
        encoder_label_test = encoder.fit_transform(Y_test)
        dummy_Y_test = np_utils.to_categorical(encoder_label_test)


        # 二分类

        def train_text():
            input = Input(shape=(w2v_dim,))
            x = Dense(50, input_dim=w2v_dim, activation='relu')(input)
            x = Dropout(0.4)(x)
            x = Dense(10, activation='relu')(x)
            y = Dense(label_categories, activation='sigmoid', name='text_out')(x)
            model = Model(inputs=input, outputs=y)

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


        score_list_1, confusion_matrix_1 = train_text()
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