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


# pos = pd.read_table(path.join(path.dirname(__file__),'..','data','pos.txt'), header=None, sep='\n', encoding='utf8')
# weibo_pos = pd.read_table(path.join(path.dirname(__file__),'..','data','weibo.txt'), header=None, sep='\n', encoding='utf8')
# pos = pos.append(weibo_pos, ignore_index=True)
# pos['label'] = 1
#
# neg = pd.read_table(path.join(path.dirname(__file__),'..','data','neg_sim.txt'), header=None, sep='\n', encoding='utf8')
# weibo_neg = pd.read_table(path.join(path.dirname(__file__),'..','data','weibo_neg.txt'), header=None, sep='\n', encoding='utf8')
# neg = neg.append(weibo_neg, ignore_index=True)
# neg['label'] = 0
# all_ = pos.append(neg, ignore_index=True)


print(sys.argv[1])
n_epochs = int(sys.argv[1])

pos = pd.read_table(path.join(path.dirname(__file__),'..','data','pos.txt'), header=None, sep='\n', encoding='utf8')
pos['label'] = 1
neg = pd.read_table(path.join(path.dirname(__file__),'..','data','neg.txt'), header=None, sep='\n', encoding='utf8')
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)
jd_len = len(all_)
print('len(all_) = '+str(len(all_)))

weibo_pos = pd.read_table(path.join(path.dirname(__file__),'..','data','weibo_pos.txt'), header=None, sep='\n', encoding='utf8')
weibo_pos['label'] = 1
all_ = all_.append(weibo_pos, ignore_index=True)
weibo_neg = pd.read_table(path.join(path.dirname(__file__),'..','data','weibo_neg.txt'), header=None, sep='\n', encoding='utf8')
weibo_neg['label'] = 0
all_ = all_.append(weibo_neg, ignore_index=True)
wb_len = len(all_) - jd_len
print('len(all_) = '+str(len(all_)))









stop_words = load_stop_words()



all_['words'] = all_[0].apply(lambda s: extract_cn_jd(s).split(' ')) #调用结巴分词
print(all_['words'])
w2v_model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data',
                                        'w2v_onlycn_100_c_2.bin'),binary=True, unicode_errors='ignore')
word2vec_dim = 100

maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法



content = []
for i in all_['words']:
    content.extend(i)#'收到', '少', '一本', '钱', '算啦', '这本', '宝宝', ……这样，形成了一条


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

pd.to_pickle(all_, path.join(path.dirname(__file__),'..','np_data','add_jd_words_doc2num_pd'))
pd.to_pickle(dict_index, path.join(path.dirname(__file__),'..','np_data','add_jd_word2seq_pd'))




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
np.save(path.join(path.dirname(__file__),'..','np_data','add_jd_embedding_weights'),embedding_weights)




#手动打乱数据
# idx = list(range(len(all_)))
# np.random.shuffle(idx)
# all_ = all_.loc[idx]

#按keras的输入要求来生成数据
X = np.array(list(all_['doc2num']))
Y = np.array(list(all_['label']))
Y = Y.reshape((-1,1)) #调整标签形状


np.save(path.join(path.dirname(__file__),'..','np_data','add_jd_doc2num'), X)
np.save(path.join(path.dirname(__file__),'..','np_data','add_jd_label'), Y)




from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Input
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')

#建立模型
batch_size = 512
# train_num = int(len(all_)/5*4)
train_num = jd_len


inputs = Input(shape=(maxlen,))
x = Embedding(output_dim = word2vec_dim,input_dim = n_symbols,mask_zero = True,weights = [embedding_weights],input_length=maxlen)(inputs)
x = LSTM(128,name='lstm_out')(x)
x = Dropout(0.5)(x)
x = Dense(50,activation='relu',name='dense_50')(x)
y = Dense(1,activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=y)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('fit')


# model.fit(X[:train_num], Y[:train_num], validation_split=0.2, batch_size = batch_size, epochs=36)
model.fit(X[:], Y[:], validation_data=(X[train_num:], Y[train_num:]), batch_size=batch_size,
          epochs=n_epochs, shuffle=True)


model.save_weights(path.join(path.dirname(__file__),'..','model','add_sentiment_lstm.h5'),overwrite=1)
# model.load_weights('sentiment_lstm.h5')


jd_scores = model.evaluate(X[:train_num], Y[:train_num], batch_size=batch_size)
print('jd => ', jd_scores)
wb_scores = model.evaluate(X[train_num:], Y[train_num:], batch_size=batch_size)
print('wb => ', wb_scores)


print(model.metrics_names[0]+':'+str(wb_scores[0])+' '
      +model.metrics_names[1]+':'+str(wb_scores[1])+' ')







def predict(sentence):
    words = sentence.split(' ')
    seq = doc2num(words,maxlen)
    arr = np.array(seq)
    arr = arr.reshape((1,maxlen))
    return model.predict(arr)

def predict_(sentence):
    words = extract_cn_jd(sentence).split(' ')
    seq = doc2num(words,maxlen)
    arr = np.array(seq)
    arr = arr.reshape((1,maxlen))
    return model.predict(arr)

def predict_list(sentence_list):
    sentiment_list = []
    for sentence in sentence_list:
        sentiment_list.append(predict(sentence))
    return sentiment_list



print(predict_('#农村现状#20年前还是个小孩，一到瓜果成熟的季节，三五个小伙伴去采摘林场里面的水果，过得很是开心。现在树上的水果都成鸟儿的美食，无人采摘。那个时候口渴了，随便找个田里的水就喝，水里夹杂着泥土的气息，现在直接站在田边就能闻到农药的味道。那个时候池塘是小伙伴的天堂，大家在里面游泳避暑 显示地图'))
print(predict_('刚看到个九零后MM和男朋友两个人站人行横道上死命招手拦出租……脑残到让人太无语了我在:勤学路 显示地图'))



text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','label_text_pd'))

print('text_all sentiment prediction')
text['text_all_senti'] = text['text_all'].apply(lambda sentence: predict(sentence))
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

text['text_single_senti'] = text['text_single'].apply(lambda sentence_list: predict_list(sentence_list))
text['single_senti_statics'] = text['text_single_senti'].apply(lambda senti_list: analyze_senti(senti_list))
senti_fea2 = np.array(list(text['single_senti_statics']))
print(senti_fea2.shape)


pd.to_pickle(text, path.join(path.dirname(__file__),'..','np_data','add_label_text_sentistatics_pd'))
# text = pd.read_pickle(path.join(path.dirname(__file__),'..','np_data','add_label_text_sentistatics_pd'))




print('extract mid level: lstm_out, dense_50')
text['text2seq'] = text['text_all'].apply(lambda sentence: doc2num(sentence.split(' '),maxlen))
print(text['text2seq'])
inputData = np.array(list(text['text2seq']))
print(inputData)

lstm_out_model = Model(inputs=model.input,outputs=model.get_layer('lstm_out').output)
lstm_out = lstm_out_model.predict(inputData)

dense_out_model = Model(inputs=model.input,outputs=model.get_layer('dense_50').output)
dense_out = dense_out_model.predict(inputData)

senti = np.concatenate((senti_fea1, senti_fea2, lstm_out, dense_out), axis = 1)
print(senti.shape)
np.save(path.join(path.dirname(__file__),'..','np_data','add_vectors_senti.npy'), senti)
print('end')



