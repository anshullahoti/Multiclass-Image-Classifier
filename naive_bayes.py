from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plot
import math
import sys


file = open(sys.argv[1],'rt')
lines = file.readlines()
file.close()

lines = np.sort(lines)
train_list = [line[:21] for line in lines]
trainDict = dict()

for i in range(0, len(lines)):
    img,label = lines[i].split()
    trainDict[label] = i

train_im = []

def pca(X):   
   [U,S,V]= np.linalg.svd(X)
   V=np.transpose(V)
   return [V]

for img in train_list:
    img = (np.array(Image.open(img).convert('L').resize((64,64),Image.ANTIALIAS)).ravel())
    train_im.append(img)

train_im = np.array(train_im)

#print(train_im)

train_mean = np.mean(train_im, axis=0)
train_im = train_im - train_mean

num_components=32
[V]=pca(train_im)

train_Coeff = np.matmul(train_im,V)
train_Coeff = train_Coeff[:, :num_components]

#print(train_Coeff)
mean_val = dict()
var_val = dict()
s=0
data = sorted(trainDict, key=trainDict.get)
for label in data:
    e = trainDict[label]+1
    p= train_Coeff[s:e, :]
    mean_val[label] = np.mean(p, axis=0)
    var_val[label]  =  np.var(p, axis=0)
    s = e

file = open(sys.argv[2],'rt')
lines = file.readlines()
file.close()
testlist = [line[:21] for line in lines]

test_im = []
for img in testlist:
    img = (np.array(Image.open(img).convert('L').resize((64,64),Image.ANTIALIAS)).ravel())
    test_im.append(img)    
     
test_im = np.array(test_im)
#print(test_im)
test_im = test_im - train_mean
test_coeff = np.matmul(test_im, V)
test_coeff = test_coeff[:, :num_components]
#print(test_coeff)
prob_val = dict()

for sample in test_coeff:
    data = sorted(trainDict, key=trainDict.get)
    for label in data:
        A = np.sqrt(2*np.pi*var_val[label])
        B= (sample-mean_val[label])
        C = 2*var_val[label]
        cal = np.array((B**2)/(C), np.dtype('c16'))
        cal = -1*cal
        prob = np.array((np.e**(cal))/A,np.dtype('c16'))
        prob_val[label] = np.prod(prob)
        #print(prob_val[label])
    print(sorted(prob_val, key=prob_val.get, reverse=True)[0])  