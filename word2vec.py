import gensim
import pickle
#import sys
#if (len(sys.argv)!=2):
#    print 'word2vec <threshold>'

o_filename = 'training_data.txt'
with open(o_filename, 'r') as f:
    sentences = f.readlines()
with open('pure_train.txt', 'r') as f:
    train = f.readlines()
with open('pure_test.txt', 'r') as f:
    test = f.readlines()

# --------------1toN MAP------------------------
# get frequent words
#threshold = int(sys.argv[1])
threshold = 10
train_word = {}
test_word = {}
for i in range(len(test)):
    q = 0
    if i % 5 == 0:
        q = 1
    token = test[i].split(' ')
    for t in token:
        if t == '':
            continue
        if not t in test_word:
            test_word[t] = 1
        else:
            if q == 1:
                test_word[t] += 1
print 'testing word:', len(test_word)
for w in train:
    w = w.strip('\n')
    if w not in train_word:
        train_word[w] = 1
    else:
        train_word[w] += 1
print 'raw training word:', len(train_word)
word2N = {} # word2N['cow'] = 23
N2word = {} # N2word[23] = 'cow'
idx = 0
# add word from testing data
for w in test_word:
    word2N[w] = idx
    N2word[idx] = w
    ++ idx
    
# filter by freq > threshold
for w, v in train_word.iteritems():
    if v > threshold and w not in test_word:
        pick.append(w)
        word2N[w] = idx
        N2word[idx] = w
        ++ idx
    elif v > threshold:
        duplicate += 1
print 'duplicate word:', duplicate
pickle.dump( word2N, open( "word2N.p", "wb" ) )
pickle.dump( N2word, open( "N2word.p", "wb" ) )

for i in range(len(sentences)):
    sentences[i] = sentences[i].strip('\n')
    token = sentences[i].split(' ')
    for t in token:
        if t not in pick:
            t = 'dummmmmy'
print 'start training model'
model = gensim.models.Word2Vec( s.split(' ') for s in sentences, size=100, window=5, min_count=1)
#model.save('th_%d.model' % (threshold))
#model.train(sentences)
#model = gensim.models.Word2Vec.load('th_10.model')
print model['you']
