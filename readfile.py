import os
import ntpath



def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_file_name(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for f in files:
            filelist.append(os.path.join(root, f))
            with open(os.path.join(root, f)) as openfile:
                for line in openfile:
                    for word in line.split():
                        print word
                
    return filelist

def main():
    txt_path = "/tmp2/weitang114/Holmes_Training_Data/training/"
    file_list = get_file_name(txt_path)

if __name__ == "__main__":
    main()
