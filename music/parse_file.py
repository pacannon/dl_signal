'''
*    Title: Introduction
*    Author: John Thickstun
*    Date: 2017
*    Code version: N/A
*    Availability: https://homes.cs.washington.edu/~thickstn/intro.html
'''
#This file creates a .npy for each music piece
import numpy as np                                       # fast vectors and matrices
from scipy import fft                                    # fast fourier transform
from intervaltree import Interval,IntervalTree
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cv_transformer', action='store_true',
    help='Use the train/valid/test split from the https://arxiv.org/pdf/2306.09827v1.pdf paper.')

args = parser.parse_args()

fs = 11000            # samples/second
window_size = 4096    # fourier window size
d = 2048              # number of features
m = 128               # (USED BY DCN) number of distinct notes
stride = 512         # samples between windows
stride_test = 128            # stride in test set
k = 64            # number of input samples per window
k_test = 64
data = np.load(open('musicnet_11khz.npz','rb'), encoding='latin1', allow_pickle=False)

# split our dataset into train and test
test_data = ['2303','2382','1819']
# Adding same validation data used by RSE
# https://github.com/LUMII-Syslab/RSE/blob/master/musicnet_data/parse_file.py#L31
validation_data = ['2131', '2384', '1792', '2514', '2567', '1876']
train_data = [ID for ID in data.files if ID not in (test_data + validation_data)]

if args.cv_transformer:
    valid_split = 0.05
    test_split = 0.1

    valids = int(valid_split * len(data.files))
    tests = int(test_split * len(data.files))

    random.seed(42)
    validation_data = random.sample(data.files, valids)
    test_data = random.sample([x for x in list(data.files) if x not in validation_data], tests)

    train_data = [i for i in data.files if i not in validation_data + test_data]

index = 0
# create the train set
for i in range(len(train_data)):
    print(i)
    X,Y = data[train_data[i]]
    for p in range(int((len(X)-window_size)/stride/k)):
        Xtrain = np.empty([k,d,2])
        Ytrain = np.zeros([k,m])
        for j in range(k):
            s = j*stride+p*k*stride# start from one second to give us some wiggle room for larger segments
            X_fft = fft.fft(X[s:s+window_size])
            Xtrain[j, :, 0] = X_fft[0:d].real
            Xtrain[j, :, 1] = X_fft[0:d].imag
            # label stuff that's on in the center of the window
            for label in Y[s+(window_size/2)]:
                if (label.data[1]) >= m:
                    continue
                else:
                    Ytrain[j,label.data[1]] = 1
        Xtrain = Xtrain.reshape(k, d*2, order='F')
        np.save("music_train_x_64_{}.npy".format(index), Xtrain.astype(np.float32))
        np.save("music_train_y_64_{}.npy".format(index), Ytrain.astype(np.float32))
        index = index + 1

# create the validation set
index = 0
for i in range(len(validation_data)):
    print(i)
    X,Y = data[validation_data[i]]
    for p in range(int((len(X)-window_size)/stride_test/k_test)):
        Xval = np.empty([k_test,d,2])
        Yval = np.zeros([k_test,m])
        for j in range(k_test):
            s = j*stride_test+p*k_test*stride_test# start from one second to give us some wiggle room for larger segments
            X_fft = fft.fft(X[s:s+window_size])
            Xval[j, :, 0] = X_fft[0:d].real
            Xval[j, :, 1] = X_fft[0:d].imag           
            # label stuff that's on in the center of the window
            for label in Y[s+d/2]:
                if (label.data[1]) >= m:
                    continue
                else:
                    Yval[j,label.data[1]] = 1
        Xval = Xval.reshape(k_test, d*2, order='F')
        np.save("music_validation_x_64_{}.npy".format(index), Xval.astype(np.float32))
        np.save("music_validation_y_64_{}.npy".format(index), Yval.astype(np.float32))
        index = index + 1




# create the test set
index = 0
for i in range(len(test_data)):
    print(i)
    X,Y = data[test_data[i]]
    for p in range(int((len(X)-window_size)/stride_test/k_test)):
        Xtest = np.empty([k_test,d,2])
        Ytest = np.zeros([k_test,m])
        for j in range(k_test):
            s = j*stride_test+p*k_test*stride_test# start from one second to give us some wiggle room for larger segments
            X_fft = fft.fft(X[s:s+window_size])
            Xtest[j, :, 0] = X_fft[0:d].real
            Xtest[j, :, 1] = X_fft[0:d].imag           
            # label stuff that's on in the center of the window
            for label in Y[s+d/2]:
                if (label.data[1]) >= m:
                    continue
                else:
                    Ytest[j,label.data[1]] = 1
        Xtest = Xtest.reshape(k_test, d*2, order='F')
        np.save("music_test_x_64_{}.npy".format(index), Xtest.astype(np.float32))
        np.save("music_test_y_64_{}.npy".format(index), Ytest.astype(np.float32))
        index = index + 1
print("finished")
