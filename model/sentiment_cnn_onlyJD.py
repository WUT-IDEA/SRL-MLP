# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import jieba
from os import path
import sys
from gensim.models import KeyedVectors
from code_pre.my_modules import load_stop_words,extract_cn,extract_cn_jd

## lstm的情感分析模型
## 输入：pos.txt, neg.txt, weibo_sim.txt, weibo_neg.txt, w2v_onlycn_100_c_2.bin
## 输出：add_jd_words_doc2num_pd, add_jd_word2seq_pd, add_jd_embedding_weights.npy
## 辅助输出：add_jd_doc2num.npy, add_jd_label.npy(直接输入到lstm的单词序号序列，及其对应的label，ps：这个是打乱顺序之后的)
## 模型训练输出：add_sentiment_lstm.h5

## 后面继续用这个lstm模型来预测微博的情感极性
## 输入：label_text_pd(里面有text_all,text_single)
## 输出：add_label_text_sentistatics_pd, add_vectors_senti.npy


# print(sys.argv[1])
# n_epochs = int(sys.argv[1])

pos = pd.read_table(path.join(path.dirname(__file__),'..','data','pos.txt'), header=None, sep='\n', encoding='utf8')
pos['label'] = 1
neg = pd.read_table(path.join(path.dirname(__file__),'..','data','neg.txt'), header=None, sep='\n', encoding='utf8')
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)



all_['words'] = all_[0].apply(lambda s: extract_cn_jd(s).split(' ')) #调用结巴分词
print(all_['words'])
w2v_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                        'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')
word2vec_dim = 100

maxlen = 100 #截断词数

content = []
for word_list in all_['words']:
    for word in word_list:
        try:
            vec = w2v_model[word]
            content.append(word)
        except BaseException as e:
            pass


def doc2matrix(s, maxlen):
    s = [i for i in s if i in content]
    matrix = np.zeros((maxlen,word2vec_dim))
    for i in range(maxlen):
        if i < len(s):
            matrix[i, :] = w2v_model[s[i]]
        else:
            matrix[i, :] = np.zeros((word2vec_dim))
    return matrix


all_['doc2matrix'] = all_['words'].apply(lambda s: doc2matrix(s, maxlen))




#手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#按keras的输入要求来生成数据
X = np.array(list(all_['doc2matrix']))
X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))
Y = np.array(list(all_['label']))
Y = Y.reshape((-1,1)) #调整标签形状


# np.save(path.join(path.dirname(__file__),'..','np_data','smote_onlyJD_jd_doc2matrix'), X)
# np.save(path.join(path.dirname(__file__),'..','np_data','smote_onlyJD_jd_label'), Y)




from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Input, Conv2D, AveragePooling2D, Flatten
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')

#建立模型
batch_size = 64
train_num = int(len(all_)/5*4)

def train(n_epochs):
    inputs = Input(shape=(maxlen,word2vec_dim,1))
    x = Conv2D(filters = 32, kernel_size = (5,100), activation='relu', name='conv_out')(inputs)
    x = AveragePooling2D((4,1),name='pool_out')(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu', name='dense_50')(x)
    y = Dense(1, activation='sigmoid', name='final_out')(x)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()
    print('fit')

    model.fit(X[:train_num], Y[:train_num], validation_split=0.2, batch_size = batch_size,
          epochs=n_epochs)


    # model.save_weights(path.join(path.dirname(__file__),'..','model','smote_onlyJD_sentiment_lstm.h5'),overwrite=1)
    # model.load_weights('sentiment_lstm.h5')


    jd_scores = model.evaluate(X[:train_num], Y[:train_num], batch_size=batch_size)
    print('on train => ', jd_scores)
    wb_scores = model.evaluate(X[train_num:], Y[train_num:], batch_size=batch_size)
    print('on test => ', wb_scores)


    print(model.metrics_names[0]+':'+str(wb_scores[0])+' '
      +model.metrics_names[1]+':'+str(wb_scores[1])+' ')


train(4)
train(6)
train(8)
train(10)
train(12)
train(14)
train(16)
train(18)
train(20)
train(22)
train(24)
train(26)
train(28)
train(30)
train(32)
train(34)
train(36)
train(38)
train(40)





'''


def predict(sentence):
    words = sentence.split(' ')
    seq = doc2matrix(words,maxlen)
    # arr = np.array(seq)
    seq = seq.reshape((1,maxlen,word2vec_dim))
    return model.predict(seq)

def predict_(sentence):
    words = extract_cn_jd(sentence).split(' ')
    seq = doc2matrix(words,maxlen)
    arr = np.array(seq)
    arr = arr.reshape((1,maxlen,word2vec_dim))
    return model.predict(arr)

def predict_list(sentence_list):
    sentiment_list = []
    for sentence in sentence_list:
        sentiment_list.append(predict(sentence))
    return sentiment_list

def model_predict_list(word_matrix_list):
    senti_list = []
    for word_matrix in word_matrix_list:
        senti_list.append(model.predict(word_matrix.reshape(1,maxlen,word2vec_dim)))
    return senti_list



print(predict_('#农村现状#20年前还是个小孩，一到瓜果成熟的季节，三五个小伙伴去采摘林场里面的水果，过得很是开心。现在树上的水果都成鸟儿的美食，无人采摘。那个时候口渴了，随便找个田里的水就喝，水里夹杂着泥土的气息，现在直接站在田边就能闻到农药的味道。那个时候池塘是小伙伴的天堂，大家在里面游泳避暑 显示地图'))
print(predict_('刚看到个九零后MM和男朋友两个人站人行横道上死命招手拦出租……脑残到让人太无语了我在:勤学路 显示地图'))



text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','smote_new_label_vectors_pd'))
# 给text增加了'text_all_senti', 'text_single_senti', =>'single_senti_statics'



print('text_all sentiment prediction')
text['text_all_senti'] = text['user_granu_vector_sequence'].apply(lambda matrix: model.predict(matrix.reshape(1,maxlen,word2vec_dim)))
senti_fea1 = np.array(text['text_all_senti'])
senti_fea1 = senti_fea1.reshape((-1,1))
print(senti_fea1.shape)

print('text_single sentiment prediction')
def analyze_senti(list):
    senti_seq = np.array(list)
    senti_fea = np.zeros((5,))
    senti_fea[0] = np.max(senti_seq)
    senti_fea[1] = np.min(senti_seq)
    senti_fea[2] = np.mean(senti_seq)
    senti_fea[3] = np.std(senti_seq)
    senti_fea[4] = np.median(senti_seq)
    return senti_fea

text['text_single_senti'] = text['microblog_granu_vector_sequence'].apply(lambda sentence_list: model_predict_list(sentence_list))
text['single_senti_statics'] = text['text_single_senti'].apply(lambda senti_list: analyze_senti(senti_list))
senti_fea2 = np.array(list(text['single_senti_statics']))
print(senti_fea2.shape)


pd.to_pickle(text, path.join(path.dirname(__file__),'..','np_data','smote_onlyJD_label_text_sentistatics_pd'))
## text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','add_label_text_sentistatics_pd'))




print('extract mid level: lstm_out, dense_50')

inputData = np.zeros((len(text),maxlen,word2vec_dim))
for i in range(len(text)):
    inputData[i,:,:] = text['user_granu_vector_sequence'][i]

# inputData = np.array(text['user_granu_vector_sequence']).reshape((len(text),maxlen,word2vec_dim))
print(inputData)

lstm_out_model = Model(inputs=model.input,outputs=model.get_layer('lstm_out').output)
lstm_out = lstm_out_model.predict(inputData)

dense_out_model = Model(inputs=model.input,outputs=model.get_layer('dense_50').output)
dense_out = dense_out_model.predict(inputData)

senti = np.concatenate((senti_fea1, senti_fea2, lstm_out, dense_out), axis = 1)
print(senti.shape)
np.save(path.join(path.dirname(__file__),'..','np_data','smote_onlyJD_vectors_senti.npy'), senti)
print('end')



'''