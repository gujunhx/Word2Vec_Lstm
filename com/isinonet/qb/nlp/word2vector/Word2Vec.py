import gensim
from gensim.models import word2vec
# wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)
sentences = word2vec.LineSentence("E:\python_workspaces\examp\data\preProcess\wordEmbdiing.txt")
a = list(sentences)
len(a)
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)
model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True)