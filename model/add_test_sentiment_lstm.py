# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import jieba
from os import path
from gensim.models import KeyedVectors

from code_pre.my_modules import load_stop_words,extract_cn_jd,extract_cn


pos = pd.read_table(path.join(path.dirname(__file__),'..','data','pos.txt'), header=None, sep='\n', encoding='utf8')
pos['label'] = 1
neg = pd.read_table(path.join(path.dirname(__file__),'..','data','neg.txt'), header=None, sep='\n', encoding='utf8')
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)
# jd_len = len(all_)
# print('len(all_) = '+str(len(all_)))
#
# weibo_pos = pd.read_table(path.join(path.dirname(__file__),'..','data','weibo_pos.txt'), header=None, sep='\n', encoding='utf8')
# weibo_pos['label'] = 1
# all_ = all_.append(weibo_pos, ignore_index=True)
# weibo_neg = pd.read_table(path.join(path.dirname(__file__),'..','data','weibo_neg.txt'), header=None, sep='\n', encoding='utf8')
# weibo_neg['label'] = 0
# all_ = all_.append(weibo_neg, ignore_index=True)
# wb_len = len(all_) - jd_len
# print('len(all_) = '+str(len(all_)))


stop_words = load_stop_words()




all_['words'] = all_[0].apply(lambda s: extract_cn_jd(s).split(' ')) #调用结巴分词
print(all_['words'])
w2v_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                        'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')
word2vec_dim = 100

maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法

# content = []
# with open(path.join(path.dirname(__file__),'..','data','status_big_seg.txt'),'r') as status_big:
#     for line in status_big:
#         word_list = line.split(' ')
#         for word in word_list:
#             content.extend(word)

content = []
for i in all_['words']:
    content.extend(i)#'收到', '少', '一本', '钱', '算啦', '这本', '宝宝', ……这样，形成了一条

# print(content)#'收到', '少', '一本', '钱', '算啦', '这本', '宝宝', ……这样，形成了一条


# 建字典索引，这是用status_big_seg.txt做的，是完全的所有的单词，{单词: 索引数字}
dict_index = pd.Series(content).value_counts()#index是词,value是数值

dict_index = dict_index[dict_index >= min_count]#这个是去掉了出现次数少于5的词
dict_index[:] = range(1, len(dict_index)+1)#对value重排了，按照1到13212排
dict_index[''] = 0 #添加空字符串用来补全，在最后加入了一个index为''，value为0的项

word_set = set(dict_index.index)



def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    # print(abc[s])#输出的是一系列词和词对应的编码序号
    # print(list(abc[s]))#输出的是[3,6128,2168,……]这样的序列，如果词不够100，则后面跟的全是0
    return list(dict_index[s])


all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))

# pd.to_pickle(all_, path.join(path.dirname(__file__),'..','np_data','add_jd_words_doc2num_pd'))
# pd.to_pickle(abc, path.join(path.dirname(__file__),'..','np_data','add_jd_word2seq_pd'))



print(u"Setting up Arrays for Keras Embedding Layer...")
n_symbols = len(dict_index) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
embedding_weights = np.zeros((n_symbols, word2vec_dim))  # 创建一个n_symbols * 100的0矩阵
for word, index in dict_index.items():  # 从索引为1的词语开始，用词向量填充矩阵
    if index == 0:
        continue
    try:
        word_vec = w2v_model[word]
    except KeyError as e:
        print("word '"+word+"' not in vocabulary")
        word_vec = np.zeros((1, word2vec_dim))
    finally:
        embedding_weights[index, :] = word_vec  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）


print(embedding_weights.shape)
# np.save(path.join(path.dirname(__file__),'..','np_data','add_jd_embedding_weights'),embedding_weights)








#手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#按keras的输入要求来生成数据
X = np.array(list(all_['doc2num']))
Y = np.array(list(all_['label']))
Y = Y.reshape((-1,1)) #调整标签形状


# np.save(path.join(path.dirname(__file__),'..','np_data','add_jd_doc2num'), X)
# np.save(path.join(path.dirname(__file__),'..','np_data','add_jd_label'), Y)




from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Input
from keras.layers import LSTM
from keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')

#建立模型
batch_size = 64
# train_num = jd_len
# train_num_4_5 = int(train_num/5*4)
train_num = int(len(all_)/5*4)

def train(n_epochs):
    inputs = Input(shape=(maxlen,))
    x = Embedding(output_dim = word2vec_dim,input_dim = n_symbols,mask_zero = True,weights = [embedding_weights],
                  input_length=maxlen)(inputs)
    x = LSTM(128,name='lstm_out')(x)
    x = Dropout(0.5)(x)
    x = Dense(50,activation='relu',name='dense_50')(x)
    y = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    print('fit')
    # model.fit(X[:train_num], Y[:train_num], validation_split=0.2, batch_size = batch_size,
    #           epochs=n_epochs, shuffle=True)
    # model.fit(X[:], Y[:], validation_split=0.2, batch_size = batch_size,
    #           epochs=n_epochs, shuffle=True)
    # model.fit(X[:], Y[:], validation_data=(X[train_num:],Y[train_num:]), batch_size = batch_size,
    #           epochs=n_epochs, shuffle=True)
    # model.fit(X[:train_num], Y[:train_num], validation_data=(X[train_num:],Y[train_num:]),
    #           batch_size = batch_size,epochs=n_epochs, shuffle=True)



    # model.fit(X[:train_num], Y[:train_num], validation_split=0.2,
    #           batch_size = batch_size,epochs=n_epochs, shuffle=True)
    model.fit(X[:int(train_num/5*4)], Y[:int(train_num/5*4)], validation_split=0.2,
              batch_size = batch_size,epochs=n_epochs, shuffle=True)
    # model.fit(X[:train_num], Y[:train_num], validation_data=(X[train_num:],Y[train_num:]),
    #           batch_size = batch_size,epochs=n_epochs, shuffle=True)
    # model.fit(X[:int((jd_len+wb_len)/5*4)], Y[:int((jd_len+wb_len)/5*4)], validation_split=0.2,
    #           batch_size = batch_size, epochs=n_epochs, shuffle=True)
    # model.fit(X[:], Y[:], validation_split=0.2,
    #           batch_size = batch_size, epochs=n_epochs, shuffle=True)


    # model.save_weights(path.join(path.dirname(__file__),'..','model','add_sentiment_lstm.h5'),overwrite=1)
    # model.load_weights('sentiment_lstm.h5')

    print('evaluate: '+str(n_epochs))
    # jd_scores = model.evaluate(X[:train_num], Y[:train_num], batch_size=batch_size)
    # print('jd => ',jd_scores)

    # wb_scores = model.evaluate(X[train_num:], Y[train_num:], batch_size = batch_size)
    # print('wb => ',wb_scores)
    wb_scores = model.evaluate(X[int(train_num/5*4):train_num], Y[int(train_num/5*4):train_num], batch_size = batch_size)
    print('wb => ',wb_scores)
    # wb_scores = model.evaluate(X[int((jd_len+wb_len)/5*4):], Y[int((jd_len+wb_len)/5*4):], batch_size = batch_size)
    # print('wb => ',wb_scores)
    # wb_scores = model.evaluate(X[train_num:], Y[train_num:], batch_size = batch_size)
    # print('wb => ',wb_scores)


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
    seq = doc2num(words,maxlen)
    arr = np.array(seq)
    arr = arr.reshape((1,maxlen))
    return model.predict(arr)


def predict_(sentence):
    words = seg_(sentence,stop_words,stop_symbols).split(' ')
    seq = doc2num(words,maxlen)
    arr = np.array(seq)
    arr = arr.reshape((1,maxlen))
    return model.predict(arr)



def predict_list(sentence_list):
    sentiment_list = []
    for sentence in sentence_list:
        sentiment_list.append(predict(sentence))
    return sentiment_list



print(predict_('我喜欢这里'))
print(predict_('我讨厌这里'))



text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_pd'))
print('h')
text['text_all_senti'] = text['text_all'].apply(lambda sentence: predict(sentence))
text_senti = np.array(text['text_all_senti'])
text_senti = text_senti.reshape((-1,1))
print(text_senti.shape)
print('hh')


def analyze_senti(list):
    sum = 0
    for i in range(len(list)):
        sum = sum + list[i]
    rate = sum/len(list)
    print(rate)
    return rate
text['text_single_senti'] = text['text_single'].apply(lambda sentence_list: predict_list(sentence_list))
text['pos_rate'] = text['text_single_senti'].apply(lambda senti_list: analyze_senti(senti_list))

pos_rate = np.array(text['pos_rate'])
pos_rate = pos_rate.reshape((-1,1))
print(pos_rate.shape)
print('hhh')

pd.to_pickle(text, path.join(path.dirname(__file__),'..','np_data','add_label_text_lstmsenti_rate_pd'))
# text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_senti_rate_pd'))

text_senti = np.array(text['text_all_senti'])
text_senti = text_senti.reshape((-1, 1))
pos_rate = np.array(text['pos_rate'])
pos_rate = pos_rate.reshape((-1, 1))



text['text2seq'] = text['text_all'].apply(lambda sentence: doc2num(sentence.split(' '),maxlen))
print(text['text2seq'])
print('hhhh')
inputData = np.array(list(text['text2seq']))
# print(inputData)


lstm_out_model = Model(inputs=model.input,outputs=model.get_layer('lstm_out').output)
lstm_out = lstm_out_model.predict(inputData)

dense_out_model = Model(inputs=model.input,outputs=model.get_layer('dense_50').output)
dense_out = dense_out_model.predict(inputData)

senti = np.concatenate((text_senti, pos_rate, lstm_out, dense_out), axis = 1)
# print(senti.shape)
np.save(path.join(path.dirname(__file__),'..','np_data','add_vectors_senti.npy'), senti)
print('end')



'''