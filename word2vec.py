import gensim
import pickle
#import sys
#if (len(sys.argv)!=2):
#    print 'word2vec <threshold>'

o_filename = 'training_data.txt'
with open(o_filename, 'r') as f:
    train = f.readlines()
with open('pure_test.txt', 'r') as f:
    test = f.readlines()

# --------------1toN MAP------------------------
# get frequent words
#threshold = int(sys.argv[1])
idx = 0
duplicate = 0
train_word = {}
word2N = {} # word2N['cow'] = 23
N2word = {} # N2word[23] = 'cow'
for i in range(len(test)):
    q = 0
    if i % 5 == 0:
        q = 1
    token = test[i].strip('\n').split(' ')
    for t in token:
        if t == '':
            continue
        if not t in train_word:
            train_word[t] = 1
            word2N[t] = idx
            N2word[idx] = t
            ++ idx
        else:
            if q == 1:
                train_word[t] += 1
for w in train:
    w = w.strip('\n')
    if w not in train_word:
        train_word[w] = 1
        word2N[w] = idx
        N2word[idx] = w
        ++ idx
    else:
        train_word[w] += 1
#        duplicate += 1
print 'training word:', len(train_word)
#print 'duplicate word:', duplicate
pickle.dump( word2N, open( "word2N.p", "wb" ) )
pickle.dump( N2word, open( "N2word.p", "wb" ) )
    
# filter by freq > threshold
threshold = 10
sentences = []
for w in train:
    w = w.strip('\n')
    if train_word[w] > threshold:
        sentences.append(w)
    else:
        sentences.append('dummmmy')
#sentences = gensim.models.doc2vec.LabeledLineSentence('training_removed.txt')
print 'start training model'
model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=1)
model.save('th_%d.model' % (threshold))
#model = gensim.models.Word2Vec.load('th_10.model')
print model['you']
