from os import path
from gensim.models import word2vec,KeyedVectors

doc = word2vec.Text8Corpus(path.join(path.dirname(__file__),'..','data','status_big_seg.txt'))
model = word2vec.Word2Vec(doc, size=100)
model.wv.save_word2vec_format(path.join(path.dirname(__file__),'..','data','w2v_onlycn_100_c_2.bin'), binary=True)
## default sg=0 => CBOW

# model = KeyedVectors.load_word2vec_format(path.join(path.dirname(__file__),'..','data','w2v_onlycn_100_c_2.bin'),
#                                           binary=True,unicode_errors='ignore')
# print(model['微博'])