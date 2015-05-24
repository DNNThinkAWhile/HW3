import gensim
import pickle

def OneofNmap(sentences, test, thresh=10):
    test = [w.strip('\n') for w in test]
    sentences = [w.strip('\n') for w in sentences]
    print 'counting...'
    freq = {}
    for w in sentences:
        token = w.split(' ')
        for t in token:
            if t not in freq:
                freq[t] = 1
            else:
                freq[t] += 1
    print 'generating map...'
    idx = 0
    word2N = {} # word2N['cow'] = 23
    N2word = {} # N2word[23] = 'cow'
    for w in sentences:
        token = w.split(' ')
        for t in token:
            if t == '':
                print 'warning'
                continue
            if freq[t] < thresh and t not in test:
                t = '-dummy-'
            if t not in word2N:
                word2N[t] = idx
                N2word[idx] = t
                ++idx
    pickle.dump( word2N, open( "word2N.p", "wb" ) )
    pickle.dump( N2word, open( "N2word.p", "wb" ) )

def wordmodel(sentences, thresh=10):
    print 'start training model...'
    model = gensim.models.Word2Vec( [s.split(' ') for s in sentences], size=100, window=5, min_count=1)
    model.save('th_%d.model' % (thresh))
    #model = gensim.models.Word2Vec.load('th_10.model')
    #print model['you']

def main():
    thresh = 10
    o_filename = 'word2vec_training_data.txt'
    sentences = open(o_filename, 'r').readlines()
    test = open('pure_test.txt', 'r').readlines()
    OneofNmap(sentences, test, thresh)
    wordmodel(sentences, thresh)

if __name__ == "__main__":
    main()
