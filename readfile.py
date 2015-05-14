import os
import ntpath
import operator


MININDEX = 0
MAXINDEX = 10000

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_file_name(path):
    d = dict()
    with open(path) as openfile:
        for line in openfile:
            for word in line.split():
                if word :
                    if word in d:
                        d[word] = d[word] + 1
                    else:
                        d[word] =  1
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse = True)
    list_d = []
    for i in range (MININDEX, MAXINDEX):
        list_d.append(sorted_d[i][0])
    print list_d
    return list_d

def main():
    txt_path = "/tmp2/weitang114/Holmes_Training_Data/training_removed.txt"
    file_list = get_file_name(txt_path)

if __name__ == "__main__":
    main()
