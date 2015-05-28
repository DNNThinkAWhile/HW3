
ans = open('answer.csv','r').readlines()
guess = open('kaggle.csv','r').readlines()

correct = 0.
total = len(ans)
for i in range(len(ans)):
    #print ans[i].strip('\n').split(',')[1]
    #print guess[i].strip('\n').split(',')[1]
    if ans[i].strip().split(',')[1] == guess[i].strip().split(',')[1]:
        print '!'
        correct = correct + 1.
    else:
        None
        #print '?', '"' + ans[i].strip().split(',')[1] + '"', '"' +  guess[i].strip().split(',')[1] + '"'
print "Score: " + str(correct/total)
