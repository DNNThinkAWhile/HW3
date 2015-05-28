import gensim
import pickle

def OneofNmap(sentences, test, thresh=10):
    print 'counting...'
    freq = {}
    for w in range(len(sentences)):
        for t in range(len(sentences[w])):
            if sentences[w][t] not in freq:
                freq[sentences[w][t]] = 1
            else:
                freq[sentences[w][t]] += 1
    print 'generating map...'
    idx = 0
    word2N = {} # word2N['cow'] = 23
    N2word = {} # N2word[23] = 'cow'
    for w in range(len(sentences)):
        for t in range(len(sentences[w])):
            if freq[sentences[w][t]] < thresh and sentences[w][t] not in test:
                sentences[w][t] = '-dummy-'
            if sentences[w][t] not in word2N:
                word2N[sentences[w][t]] = idx
                N2word[idx] = sentences[w][t]
                idx = idx + 1
    pickle.dump( word2N, open( "word2N.p", "wb" ) )
    pickle.dump( N2word, open( "N2word.p", "wb" ) )

def wordmodel(sentences, thresh=10):
    print 'start training model...'
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1)
    model.save('th_%d.model' % (thresh))
    #model = gensim.models.Word2Vec.load('th_10.model')
    #print model['you']

def main():
    thresh = 10
    o_filename = 'word2vec_training_data.txt'
    sentences = open(o_filename, 'r').readlines()
    test = open('pure_test.txt', 'r').readlines()
    for i in range(len(test)):
        test[i] = test[i].strip('\n')
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip('\n')
        sentences[i] = sentences[i].split(' ')
    OneofNmap(sentences, test, thresh)
    wordmodel(sentences, thresh)

if __name__ == "__main__":
    main()
