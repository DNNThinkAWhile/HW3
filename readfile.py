import os
import ntpath



def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_file_name(path):
    filename = []
    for root, dirs, files in os.walk(path):
        for f in files:
            filename.append(path_leaf(f))
    return filename

def main():
    txt_path = "/tmp2/weitang114/Holmes_Training_Data/training/"
    file_list = get_file_name(txt_path)
    i = 0
    for f in file_list:
        i = i + 1
        print f
        print i

if __name__ == "__main__":
    main()
