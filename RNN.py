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

word2N = pickle.load(open( "word2N.p", "rb" )) 
print len(word2N)
train = open('training_data.txt').readlines()[0]
train = train.split(' ')

ptr = 0
instance_x = [None]*4 # word2vec
instance_y = [None]*4 # 1ofN ans
x = []
ans_y = []

for i in range(len(train)):
    if i+3 > len(train):
        break
    for j in range(4):
        try:
            instance_x[j] = model[train[i+j]]
        except:
            instance_x[j] = model['-dummy-']
        try: 
            n = word2N[train[i+j+1]]
        except:
            n = word2N['-dummy-']
        instance_y[j] = np.zeros(len(word2N))
        instance_y[j][n] = 1
    x.append(instance_x)
    ans_y.append(instance_y)
    if (i+1) % batch == 0:
        print i
        x = np.array(x)
        ans_y = np.array(ans_y)
        x = np.swapaxes(x,0,1)
        ans_y = np.swapaxes(ans_y,0,1)
        loss,wi,wh,wo,grad = forward.forward(x, ans_y)
        x = []
        ans_y = []
        print 'loss: ',loss
        if (i+1) % (100*batch) == 0:
            print 'saving model'
            np.save('wi_iter_'+str(i+1), wi)
            np.save('wh_iter_'+str(i+1), wh)
            np.save('wo_iter_'+str(i+1), wo)
