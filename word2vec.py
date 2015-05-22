import gensim
import pickle
#import sys
#if (len(sys.argv)!=2):
#    print 'word2vec <threshold>'

o_filename = 'training_data.txt'
with open(o_filename, 'r') as f:
    sentences = f.readlines()
#with open('pure_train.txt', 'r') as f:
#    train = f.readlines()
with open('pure_test.txt', 'r') as f:
    test = f.readlines()

# --------------1toN MAP------------------------
# get frequent words
#threshold = int(sys.argv[1])
threshold = 10
idx = 0
train_word = {}
word2N = {} # word2N['cow'] = 23
N2word = {} # N2word[23] = 'cow'
for w in sentences:
    token = w.strip('\n').split(' ')
    for t in token:
        if t == '':
            continue
        if not t in train_word:
            train_word[t] = 1
            word2N[t] = idx
            N2word[idx] = t
            ++ idx
        else:
            train_word[t] += 1

for w in sentences:
    token = w.strip('\n').split(' ')
    for t in token:
        if t == '':
            continue
        if t not in train_word:
            train_word[t] = 1
            word2N[t] = idx
            N2word[idx] = t
            ++ idx
        else:
            train_word[t] += 1

print 'training word:', len(train_word)
pickle.dump( word2N, open( "word2N.p", "wb" ) )
pickle.dump( N2word, open( "N2word.p", "wb" ) )

for w in sentences:
    w = w.strip('\n')
    token = w.split(' ')
    for t in token:
        if t == '':
            continue
        if train_word[t] < threshold:
            t = 'dummmmmy'

print sentences
print 'start training model'
model = gensim.models.Word2Vec( [s.split(' ') for s in sentences], size=100, window=5, min_count=1)
model.save('th_%d.model' % (threshold))
#model = gensim.models.Word2Vec.load('th_10.model')
print model['you']
