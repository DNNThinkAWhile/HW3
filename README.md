# HW3 RNNLM/LSTM

Environment: Linux 3.16.0-4-amd64 #1 SMP Debian 3.16.7-ckt2-1 (2014-12-08) x86_64 GNU/Linux
GPU: Tesla K40m

Training Data Preprocessing
    usage: python preprocess.py

Word2Vec model
    usage: python word2vec.py <threshold>
    @ parameter description
        threshold - frequency threshold

Train
    usage: python RNN.py <batch> <threshold>      
    @ parameter description
        batch - batch size
        threshold - threshold that you uesd in building Word2Vec model

Testing Data Preprocessing
    1) cut_test.py
        usage: python cut_tests.py
    2) test.sh
        usage: ./test.sh
    3) test_dummy.py
        usage: python test_dummy.py <threshold>
        threshold - threshold that you uesd in building Word2Vec model

Predict
    usage: python RNNout.py
        since we define Theano function and variable in the global section, you need to load the desired model into predict.py manually.

The original training/testing data need to be in folder 'Holmes_Training_Data/training' and 'Holmes_Training_Data/testing' respectively.
After training data preprocessing , there will be 2 main data: training_data.txt and word2vec_training_data.txt, the former one is used in Train progress, and the latter is used for training word2vec model.
After testing data preprocessing, there will be several directories made in Holmes_Training_Data/test/, containing different formats of testing data.
Word2Vec model will be named 'th_<threshold>.model' and the final RNN model will generate outfile 'kaggle.csv'  
