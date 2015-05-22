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
#threshold = sys.argv[1]
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

train_word_star = []
# add word from testing data
for w in test_word:
    train_word_star.append(w)
    
# filter by freq > threshold
duplicate = 0
for w, v in train_word.iteritems():
    if v > threshold and w not in test_word:
        train_word_star.append(w)
    elif v > threshold:
        duplicate += 1
print 'training word:', len(train_word_star)
print 'duplicate word:', duplicate
