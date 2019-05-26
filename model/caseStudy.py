# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os

from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


# text = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..','np_data','label_text_pd'))
# print(text)
#
# vec = np.load(os.path.join(os.path.dirname(__file__), '..','np_data','vectors_label_textvec.npy'))
# print(vec)
# print(vec.shape)
# w2v_dim = 100
#
# sample_size = vec.shape[0]
# X_train = vec[:int(sample_size * 0.8), 3:]
# Y_train = vec[:int(sample_size * 0.8), 0]
# X_test = vec[int(sample_size * 0.8):, 3:]
# Y_test = vec[int(sample_size * 0.8):, 0]
#
#
# encoder = LabelEncoder()
# encoder_label_train = encoder.fit_transform(Y_train)
# dummy_Y_train = np_utils.to_categorical(encoder_label_train)
# encoder_label_test = encoder.fit_transform(Y_test)
# dummy_Y_test = np_utils.to_categorical(encoder_label_test)
#
#
# input = Input(shape=(w2v_dim,))
# x = Dense(50, input_dim=w2v_dim, activation='relu')(input)
# x = Dropout(0.4)(x)
# x = Dense(10, activation='relu')(x)
# y = Dense(2, activation='sigmoid', name='text_out')(x)
# model = Model(inputs=input, outputs=y)
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# history = model.fit(X_train, dummy_Y_train, shuffle=True, epochs=20, batch_size=64,
#           validation_split=0.2, verbose=2)
# history_dic = history.history
# for key in history_dic.keys():
#     print(key,history_dic[key])
#
# predictions = model.predict(X_test)
# print(predictions.shape, text['text_all'][int(sample_size * 0.8):].shape, text[1][int(sample_size * 0.8):].shape, )
#
#
# text['text_all'][int(sample_size * 0.8):].to_csv('../data/caseStudy_text.csv', encoding='utf8')
#
# result = pd.DataFrame({'label': text[1][int(sample_size * 0.8):],
#                        '0_prob': predictions[:, 0],
#                        '1_prob': predictions[:, 1]})
# result['label_pred'] = 0
# for i in result.index:
#     if result['1_prob'][i] > result['0_prob'][i]:
#         result['label_pred'][i] = 1
#
# print(result)
# result.to_csv('../data/caseStudy_pred.csv', encoding='utf8')










'''
sentiment
'''



# smote_vec = np.load(os.path.join(os.path.dirname(__file__), '..', 'np_data', 'smote_user_label_vectors.npy'))
# smote_senti = np.load(os.path.join(os.path.dirname(__file__), '..', 'np_data', 'smote_simJD_vectors_senti.npy'))
# print(smote_vec.shape)
# print(smote_senti.shape)
# smote_senti = smote_senti[:, 6:134]
#
# data = np.concatenate([smote_vec, smote_senti], axis=1)
#
# test = data[2510:3138,:]
# train_1 = data[:2510, :]
# train_2 = data[3138:, :]
# train = np.concatenate([train_1, train_2], axis=0)
#
# print('train_shape =', train.shape)
# print('test_shape =', test.shape)
# X_train_text = train[:, 3: 103]
# X_test_text = test[:, 3: 103]
# X_train_lstm = train[:, 103: 231]
# X_test_lstm = test[:, 103: 231]
# Y_train = train[:, 0]
# Y_test = test[:, 0]
#
# encoder = LabelEncoder()
# encoder_label_train = encoder.fit_transform(Y_train)
# dummy_Y_train = np_utils.to_categorical(encoder_label_train)
# encoder_label_test = encoder.fit_transform(Y_test)
# dummy_Y_test = np_utils.to_categorical(encoder_label_test)
#
#
# input_text = Input(shape=(100,))
# input_lstm = Input(shape=(128,))
#
# text_out = Dense(50, input_dim=100, activation='relu')(input_text)
# text_out = Dropout(0.4)(text_out)
# text_out = Dense(10, activation='relu')(text_out)
# text_out = Dropout(0.4)(text_out)
# # auxi_out = Dense(label_categories, activation='sigmoid', name='auxi_out')(auxi_out)
#
# lstm_out = Dense(50, input_dim=128, activation='relu')(input_lstm)
# lstm_out = Dropout(0.4)(lstm_out)
# lstm_out = Dense(10, activation='relu')(lstm_out)
# lstm_out = Dropout(0.4)(lstm_out)
# # senti_out = Dense(label_categories, activation='sigmoid', name='senti_out')(senti_out)
#
# x = concatenate([text_out, lstm_out])
# # x = Dense(10, activation='relu')(x)
# y = Dense(2, activation='sigmoid')(x)
#
# model = Model(inputs=[input_text, input_lstm], outputs=y)
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit([X_train_text, X_train_lstm], dummy_Y_train,
#           shuffle=True, validation_split=0.2, epochs=100, batch_size=64, verbose=2)
# history_dic = history.history
# for key in history_dic.keys():
#     print(key,history_dic[key])
#
# score = model.evaluate([X_test_text, X_test_lstm], dummy_Y_test, batch_size=64)
# print('score =',score)
# predictions = model.predict([X_test_text, X_test_lstm])
#
# result = pd.read_csv('../data/caseStudy_pred.csv')
# result['addlstm_0_prob'] = predictions[:, 0]
# result['addlstm_1_prob'] = predictions[:, 1]
# result['addlstm_label_pred'] = 0
# for i in result.index:
#     if result['addlstm_1_prob'][i] > result['addlstm_0_prob'][i]:
#         result['addlstm_label_pred'][i] = 1
#
# print(result)
# result.to_csv('../data/caseStudy_addlstm.csv', encoding='utf8')







'''
result
'''

text = pd.read_csv('../data/caseStudy_text.csv', delimiter=',', header=None)
result = pd.read_csv('../data/caseStudy_addlstm.csv',delimiter=',')

text_list = []
for i in range(len(result)):
    if result['label'][i] != result['label_pred'][i]:
        text_list.append(str(result['label'][i]))
        text_list.append(text[1][i])

with open('../data/caseStudy_selectedText.txt', encoding='utf8', mode='w') as f:
    for t in text_list:
        f.write(t)
        f.write('\n')
