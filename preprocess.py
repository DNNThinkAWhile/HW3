import os

# --------------TRAINING FILE--------------------
# remove header till *END*
for i_filename in os.listdir('Holmes_Training_Data'):
    cmd = "cat Holmes_Training_Data/%s | LC_ALL=C sed -n '/*END*/,$p' | sed '1d' > training/%s" % (i_filename, i_filename)
    os.system(cmd)

# remove char based on restriction in processing.sh
os.system("./preprocessing.sh")    

# remove Footnote and the end of the file
o_filename = 'training_data.txt'
fout = open('tmp2.txt', 'w')
with open('tmp.txt', 'r') as f:
    raw = f.readlines()
flag = 0
flag_intext = 0
for line in raw:
    if line.startswith('End of The Project Gutenberg'):
        continue
    if line.startswith('Footnote'):
        flag = 1
        continue
    elif 'Footnote' in line:
        flag_intext = 1 
        fout.write(line.split('Footnote')[0])
        continue
    if line == '\n':
        flag = 0
        flag_intext = 0
        fout.write('\n')
    elif not flag and not flag_intext:
        fout.write(line)

fout.flush()
#cmd = "cat %s | sed -e 's/ /\\'$'\\n/g' | sed '/^ *$/d' > %s " % ('tmp2.txt', o_filename)
cmd = "cat %s | sed '/^ *$/d' > %s " % ('tmp2.txt', o_filename)
os.system(cmd)

with open('tmp.txt', 'r') as f:
    print 'tmp', len(f.readlines())
with open('tmp2.txt', 'r') as f:
    print 'before', len(f.readlines())
with open(o_filename, 'r') as f:
    print 'after', len(f.readlines())

# --------------TESTING FILE---------------------
# remove "1a)" and "[].'"...
os.system("./pure.sh")
