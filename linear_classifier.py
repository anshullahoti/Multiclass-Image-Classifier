from PIL import Image
import numpy as np
import os
import sys
from numpy import *

file = open(sys.argv[1],'rt')
lines = file.readlines()
file.close()
lines = np.sort(lines)

train_list=[]
p  = len(lines)
for line in lines:
    train_list.append(line[:21])

classlist=[]
for line in lines:
    a,b = line.split()
    classlist.append(b)

imgDict = dict()
j=-1
num_components=33

def pca(X):   
   [U,S,V]= np.linalg.svd(X)
   V=np.transpose(V)
   return [V]

for i in range(0, p):
    label = lines[i].split()[1]
    if label not in imgDict:
        j+=1
        imgDict[label] = j

num_class = len(imgDict)
num_train = len(train_list)

train_im = []
for img in train_list:
    img = (np.array(Image.open(img).convert('L').resize((64,64),Image.ANTIALIAS)).ravel())
    train_im.append(img)

mean_X = np.mean(train_im, axis=0)
train_im = train_im - mean_X

[V] = pca(train_im)
train_Coeff = np.matmul(train_im, V)
train_Coeff = train_Coeff[:, :num_components]
a,b = train_Coeff.shape
for i in range(a):
    train_Coeff[i][32] = 1

w = np.array(random.random((num_class, 33)))
eta = 0.00001

for i in range(0,4000):
    wtemp =np.array(zeros((num_class, 33)))

    for j in range(0,num_train):

        label = classlist[j]
        prod =  np.array(np.matmul(w, train_Coeff[j]), dtype=float64)
        large = max(prod)
        prod = np.array(prod-large, dtype=float64)
        len1 = np.array(pow(np.e,prod))
        den = np.sum(len1)
        
        prime = (np.matmul(w[imgDict[label], :], train_Coeff[j])-large)
        num = pow(np.e,prime)
        prob = num/den
        wtemp[imgDict[label], :] = np.array(add(wtemp[imgDict[label], :],(1-prob)*train_Coeff[j]))
    w = np.array(add(w, eta*wtemp))

file = open(sys.argv[2],'rt')
lines = file.readlines()
file.close()
samplelist = [line[:21] for line in lines]

smatrix = []

for img in samplelist:
    img = (np.array(Image.open(img).convert('L').resize((64,64),Image.ANTIALIAS)).ravel())
    smatrix.append(img)

smatrix = smatrix - mean_X
test_coeff = np.matmul(smatrix, V)

test_coeff = test_coeff[:, :num_components]
a, b = test_coeff.shape
for i in range(a):
    test_coeff[i][32] = 1

for sample in test_coeff:
    probMatrix   = np.matmul(w, sample)
    tp = argmax(probMatrix)
    for label, ind in imgDict.items():
        if  ind == tp:
            print(label)