import pickle
import gensim

model = gensim.models.Word2Vec.load('th_10.model')

test = open('testing/testing_clean.txt', 'r').readlines()
outfile = open('testing/testing_dummy.txt', 'w')

for line in test:
    token = line.split()
    for i in range(len(token)):
        if token[i] not in model and token[i] != '_':
            token[i] = '-dummy-'
        if i == len(token)-1:
            outfile.write(token[i])
        else:
            outfile.write(token[i] + ' ')
    outfile.write('\n')
