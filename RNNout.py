#import predict
import gensim
import numpy as np
import pickle
import csv
thresh = 10
#wi = np.load('wi_iter_' + it + '.npy')
#wh = np.load('wh_iter_' + it + '.npy')
#wo = np.load('wo_iter_' + it + '.npy')
model = gensim.models.Word2Vec.load('th_%d.model' % thresh)
print 'model loaded.'
word2N = pickle.load(open( "word2N.p", "rb" ))

test = open('testing/testing_clean.txt', 'r').readlines()
outfile = csv.writer(open('kaggle.csv', 'w'))
outfile.writerow('Id,Prediction')
x = [None]*4
a = [None]*5
c = ['a', 'b', 'c', 'd', 'e']
for i in range(0,len(test),6):
    q = test[i].strip('\n').split('_')
    maxi = 0
    max_prob = 0
    normal = 0
    print 'q',q
    for j in range(5):
        a[j] = test[i+j+1].split()[1]
        # _XXX
        print 'a',a
        if len(q[1].split()) > 3:
            x[0] = [ model[a[j]], model[q[1].split()[0]], model[q[1].split()[1]], model[q[1].split()[2]] ]
            normal += 1
        # X_XX
        if len(q[0].split()) > 1 and len(q[1].split()) > 2:
            x[1] = [ model[q[0].split()[-1]], model[a[j]], model[q[1].split()[0]], model[q[1].split()[2]] ]
            normal += 1
        # XX_X
        if len(q[0].split()) > 2 and len(q[1].split()) > 1:
            x[2] = [ model[q[0].split()[-2]], model[q[0].split()[-1]], model[a[j]], model[q[1].split()[0]] ]
            normal += 1
        # XXX_
        if len(q[0].split()) > 3:
            x[3] = [ model[q[0].split()[-3]], model[q[0].split()[-2]], model[q[0].split()[-1]], model[a[j]] ]
            normal += 1
    print x
    #y = predict(x)
    '''
    for choice in range(5):
        tmp_prob = 1
        for pattern in range(4):
            tmp_prob *= y[pattern][word2N[a[choice]]]
        tmp_prob /= normal
        if tmp_prob > max_prob:
            max_prob = tmp_prob
            maxi = choice
    sol = [str(i/5), c[choice]]
    outfile.writerow(sol)'''
