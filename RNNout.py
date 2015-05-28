import predict
import gensim
import numpy as np
import pickle
import csv
import sys

if len(sys.argv) < 1:
    print 'RNNout.py <threshold>'
    quit()
thresh = int(sys.argv[1])
model = gensim.models.Word2Vec.load('th_%d.model' % thresh)
print 'model loaded.'
word2N = pickle.load(open( "word2N.p", "rb" ))

test = open('testing/testing_dummy.txt', 'r').readlines()
outfile = csv.writer(open('kaggle.csv', 'w'))
outfile.writerow(['id','answer'])
#x = [None]*20
a = [None]*5
c = ['a', 'b', 'c', 'd', 'e']
for i in range(0,len(test),6):
    q = test[i].strip('\n').split('_')
    maxi = 0
    max_prob = 0
    normal = 0
    pattern = [[],[],[],[],[]]
    #print 'q',q
    x = []
    for j in range(5):
        a[j] = test[i+j+1].split()[1]
        # _XXX
        #print 'a',a
        if len(q[1].split()) > 3:
            #x.append(model[a[j]])
            #x.append(model[q[1].split()[0]])
            #x.append(model[q[1].split()[1]])
            #x.append(model[q[1].split()[2]])
            #x[4*j] = [ model[a[j]], model[q[1].split()[0]], model[q[1].split()[1]], model[q[1].split()[2]] ]
            x.append( [ model[a[j]], model[q[1].split()[0]], model[q[1].split()[1]], model[q[1].split()[2]] ])
            pattern[j].append(0)
            normal += 1
        # X_XX
        if len(q[0].split()) > 1 and len(q[1].split()) > 2:
            #x.append(model[q[0].split()[-1]])
            #x.append(model[a[j]])
            #x.append(model[q[1].split()[0]])
            #x.append(model[q[1].split()[2]])
            #x[4*j+1] = [ model[q[0].split()[-1]], model[a[j]], model[q[1].split()[0]], model[q[1].split()[2]] ]
            x.append([ model[q[0].split()[-1]], model[a[j]], model[q[1].split()[0]], model[q[1].split()[2]] ])
            pattern[j].append(1)
            normal += 1
        # XX_X
        if len(q[0].split()) > 2 and len(q[1].split()) > 1:
            #x[2].append(model[q[0].split()[-2]])
            #x[2].append(model[q[0].split()[-1]])
            #x[2].append(model[a[j]])
            #x[2].append(model[q[1].split()[0]])
            #x[4*j+2] = [ model[q[0].split()[-2]], model[q[0].split()[-1]], model[a[j]], model[q[1].split()[0]] ]
            x.append([ model[q[0].split()[-2]], model[q[0].split()[-1]], model[a[j]], model[q[1].split()[0]] ])
            pattern[j].append(2)
            normal += 1
        # XXX_
        if len(q[0].split()) > 3:
            #x[3].append(model[q[0].split()[-3]])
            #x[3].append(model[q[0].split()[-2]])
            #x[3].append(model[q[0].split()[-1]])
            #x[3].append(model[a[j]])
            #x[4*j+3] = [ model[q[0].split()[-3]], model[q[0].split()[-2]], model[q[0].split()[-1]], model[a[j]] ]
            x.append([ model[q[0].split()[-3]], model[q[0].split()[-2]], model[q[0].split()[-1]], model[a[j]] ])
            pattern[j].append(3)
            normal += 1
    x = np.array(x)
    x = np.swapaxes(x,0,1)
    y = predict.predict(x)
    print y.shape
    batch = 0
    for choice in range(5):
        tmp_prob = 1
        for p in range(len(pattern[choice])):
            tmp_prob *= y[p, batch, word2N[a[choice]]]
        batch = batch + 1
        tmp_prob = tmp_prob/normal
        if tmp_prob > max_prob:
            max_prob = tmp_prob
            maxi = choice
    sol = [str(i/6 + 1), c[maxi]]
    outfile.writerow(sol)
