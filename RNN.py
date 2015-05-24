import gensim
import pickle
import sys
import numpy as np
import forward

print 'usage: RNN <batch size> <freq threshold>'

batch = 128
thresh = 10
model = gensim.models.Word2Vec.load('th_%d.model' % thresh)
print 'model loaded.'

word2N = pickle.load(open( "word2N.p.p", "rb" )) 
train = open('training_data.txt').readlines()
train = train.split(' ')

ptr = 0
x = [None]*4 # word2vec
ans_y = [None]*4 # 1ofN ans
for i in range(len(train)):
    for j in range(4):
        x[j] = model[train[i]]
        y = word2N[train[i+1]]
        ans_y[j] = np.zeros(4)
        ans_y[j][y] = 1
    if i == batch:
        y,a,z = forward(x, ans_y)
        print 'success!!!!!!!!!!!!!!!!!!!'

