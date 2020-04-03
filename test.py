# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:49:01 2018

@author: utke.markus
"""

from pathlib import Path
import numpy as np
np.random.seed(7)
from scipy import stats
import imageio
import cv2
from skimage import transform,io
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction import image
from sklearn import ensemble
from sklearn import svm
import tensorflow as tf
tf.set_random_seed(9)
from keras.models import Sequential, load_model
from keras.applications import xception, densenet, resnet50, mobilenet_v2
from keras.layers import Input, Dense, Lambda, Add, Flatten, BatchNormalization, LSTM, add
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import initializers
from keras.applications import imagenet_utils
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.metrics import categorical_accuracy
import itertools
import keras.backend as K
import matplotlib.pyplot as plt
import time
import glob
from os import path
from itertools import permutations
from importData import DataSequenceNew, importDataSubjectiveFrames
from vis.visualization import visualize_cam
import pickle
from tools import buildModel, PatchPatterns

def getMetrics(yPred, y, mode = 'regression', verbose = True):
    if mode == 'regression':
        r2 = metrics.r2_score(y, yPred)
        rmse = np.sqrt(metrics.mean_squared_error(y, yPred))
        pcc = stats.pearsonr(y, yPred)[0]
        srcc = stats.spearmanr(y, yPred)[0]
        if verbose:
            print("R²: %1.4f \t RMSE: %1.4f \t PCC: %1.4f \t SRCC: %1.4f" % (r2, rmse, pcc, srcc))
        return r2, rmse, pcc, srcc
    elif mode == 'classification':
        acc = metrics.accuracy_score(np.argmax(y, axis = -1), np.argmax(yPred, axis = -1))
        if verbose:
            print("Accuracy: %1.4f" % (acc))
        return acc
    
def setTrainability(model, nTrainableLayers):
    for layer in model.layers[:-nTrainableLayers]:
        layer.trainable = False
    
    for layer in model.layers[-nTrainableLayers:]:
        layer.trainable = True


#%%

preprocessList = [xception.preprocess_input, resnet50.preprocess_input, densenet.preprocess_input, mobilenet_v2.preprocess_input]
modelBuilderList = [xception.Xception, resnet50.ResNet50, densenet.DenseNet121, mobilenet_v2.MobileNetV2]

modelNumber = 3

partModel = modelBuilderList[modelNumber](include_top = False, weights = None, pooling = 'avg', input_shape = (299,299,3))
weightModel = modelBuilderList[modelNumber](include_top = False, weights = None, pooling = 'avg', input_shape = (224,224,3))
partModel.set_weights(weightModel.get_weights())
outputs = partModel.output
outputs = Dense(1, activation = 'linear')(outputs)
model = Model(inputs = partModel.input, outputs = outputs)
model.summary()

layers = model.layers



#%%

#model.summary()

c = 0
for i in range(len(model.layers)):
    if model.layers[-i].name == "k":
        break;
    else:
        c += 1

#%%
c2 = 0
o = []
for l in layers[-11111:]:
    for w in l.get_weights():
        o.append(w)
        c2 += np.size(w)
c2
      
# densenet: 204, 4460801  
# xception:   8, 4758017
# resnet:    12, 4473857




#%%
dsn = DataSequenceNew("D:\\Dataset\\1080", [0], batchSize=25, patchSize=299, nPatches = 1, shuffle=False, preprocess=None, returnMode = 1)

test = dsn.__getitem__(0)
plt.imshow(test[0][1].astype(int))
test[1][1]

trainIndexes = np.hstack((np.arange(0, 45), np.arange(57, 81), np.arange(90,162)))
testIndexes = np.hstack((np.arange(45, 57), np.arange(81, 90)))
test = [path.basename(folder) for folder in glob.glob("D:\\Dataset\\1080\\" + "\\*")]

for i in testIndexes:
    print(test[i])

data.columns.values

#%% ###################################### nLayers ###########################################


filepath = "J:\\DataSet\\NewDataset\\All\\"


trainIndexes = np.arange(390)
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))


trainIndexes = np.delete(trainIndexes, testIndexes)

nTrainVideos = len(trainIndexes)
nTestVideos = len(testIndexes)

# Test: CSGO/2, Hearthstone/2, Dota, ProjectCar # CSGO, Hearthstone, LoL, Fifa, HeroesOfTheStorm, PU

batchSize = 64
everynthTrain = 53
everynthTest = 23

dsTest = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = everynthTest, batchSize = 100, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1)  
yTest = []
for i in range(dsTest.__len__()):
    yTest.append(dsTest.__getitem__(i))
yTest = np.concatenate(yTest)

#%%



nLayersList = [429, 376, 334, 288, 204, 116, 60, 1]


metricsListFrame = []
metricsListVideo = []
for nLayer in nLayersList: 
    for k in range(3):
        metricsListFrame.append(getMetrics(np.load("Results\\final_results\\nLayers\\denseNet%ievery53thbs16_bestEpoch%i.npy" % (nLayer,k)), yTest, verbose = False))
        metricsListVideo.append(getMetrics(np.mean(np.load("Results\\final_results\\nLayers\\denseNet%ievery53thbs16_bestEpoch%i.npy" % (nLayer,k)).reshape((len(testIndexes),-1)), axis=-1), np.mean(yTest.reshape((len(testIndexes),-1)), axis=-1), verbose=False))

metricsListFrame = np.array(metricsListFrame)
metricsListVideo = np.array(metricsListVideo)


metricsList = np.stack([metricsListFrame, metricsListVideo], axis = 1)

test = metricsList.reshape(len(nLayersList),3,2,4)
metricsList = np.mean(test, axis=1)
metricsStdError = np.std(test, axis=1, ddof=1)/np.sqrt(3)
test1 = test[:,:,:,1]
#%%

plt.rcParams.update({'font.size': 12})
fig = plt.figure()
plt.bar(np.arange(len(metricsList))*3+1, metricsList[:,0,1], align='center', alpha=0.9, width = 1, color = 'c')
plt.bar(np.arange(len(metricsList))*3, metricsList[:,1,1], align='center', alpha=0.9, width = 1, color = 'g')
plt.xticks(np.arange(len(metricsList))*3+0.5, ["7038529\n$n$ = 120", "6656769\n$n$ = 107", "6267777\n$n$ = 94", "5593601\n$n$ = 82", "4460801\n$n$ = 57", "2191361\n$n$ = 33", "1233409\n$n$ = 16", "1025\n$n$ = 0"]) #
plt.ylabel('RMSE')
plt.xlabel('Number of weights trained')
plt.legend(["Frame level", "Video level"], loc=2)
plt.ylim(0,11)

plt.savefig('Results\\numberOfLayers.pdf', bbox_inches = 'tight')


#%%
for metric in metricsList:
    ind = np.argmin(metric[0][:,1])
    print(ind)
    minMetrics.append(metric[:,ind])
    print("R²: {0:1.4f} \t RMSE: {1:1.4f} \t PCC: {2:1.4f} \t SRCC: {3:1.4f}".format(*metric[0][ind,:]))

print()

for metric in metricsList:
    ind = np.argmin(metric[1][:,1])
    print("R²: {0:1.4f} \t RMSE: {1:1.4f} \t PCC: {2:1.4f} \t SRCC: {3:1.4f}".format(*metric[1][ind,:]))

minMetrics = np.flip(np.array(minMetrics), axis = 0)
minMetrics[:,0,2]
#%%
#plt.bar([29, 53, 103, 203], minMetrics[:,0,1])
#plt.bar([29, 53, 103, 203], minMetrics[:,1,1])
plt.rcParams.update({'font.size': 12})
fig = plt.figure()
plt.bar(np.arange(len(metricsList))*3+1, minMetrics[:,0,1], align='center', alpha=0.9, width = 1, color = 'c')
plt.bar(np.arange(len(metricsList))*3, minMetrics[:,1,1], align='center', alpha=0.9, width = 1, color = 'g')
plt.xticks(np.arange(len(metricsList))*3+0.5, ["933\n$n$ = 403", "1555\n$n$ = 203", "2799\n$n$ = 103", "5287\n$n$ = 53", "9952\n$n$ = 23"])
plt.ylabel('RMSE')
plt.xlabel('Number of frames in the training set')
plt.legend(["Frame level", "Video level"], loc=1)
plt.ylim(0,9.001)

plt.savefig('Results\\numberOfSamples.pdf', bbox_inches = 'tight')

#%%


filepath1 = "J:\\DataSet\\NewDataset\\All\\"
filepath2 = "J:\\DataSet\\"


trainIndexes1 = np.arange(390)
trainIndexes2 = []# np.arange(71)
testIndexes1 = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))
testIndexes2 =  []# np.arange(0, 36)

trainIndexes1 = np.delete(trainIndexes1, testIndexes1)
trainIndexes2 = []# np.delete(trainIndexes2, testIndexes2)

nTrainVideos = len(trainIndexes1) + len(trainIndexes2)
nTestVideos = len(testIndexes1) + len(testIndexes2)

# Test: CSGO/2, Hearthstone/2, Dota, ProjectCar # CSGO, Hearthstone, LoL, Fifa, HeroesOfTheStorm, PU

batchSize = 64
everynthTrain = 53
everynthTest = 23

dsTest = DataSequenceMix(filepath1, testIndexes1, filepath2, testIndexes2, patchSize = 299, everynth = everynthTest, batchSize = 100, preprocess = xception.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1)  
yTest = []
for i in range(dsTest.__len__()):
    yTest.append(dsTest.__getitem__(i))
yTest = np.concatenate(yTest)

#%%

predsList = [58, 38, 18, 8]

for predsI in predsList:
    
    preds = np.load("Results\\xception%ievery53thNew_bestPreds.npy" % predsI)
    
    #plt.bar([29, 53, 103, 203], minMetrics[:,0,1])
    #plt.bar([29, 53, 103, 203], minMetrics[:,1,1])
    plt.rcParams.update({'font.size': 12})
#    yTest = yTest[:120]
    plt.figure(figsize = (4,4))
    plt.plot(yTest, preds, 'co', markersize = .8) 
    plt.plot(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1),np.mean(preds.reshape((nTestVideos,-1)), axis=-1), 'go', markersize = 4.25)
    plt.plot(np.arange(101), 'k')
    plt.xlim((-10,110))
    plt.ylim((-10,110))
    #plt.title("Epoch %i" % (i))
    plt.xlabel("Real VMAF")
    plt.ylabel("Predicted VMAF")
    plt.legend(["Frame level", "Video level"], loc = 2)
    
    plt.savefig('Results\\preds%i.pdf' % predsI, bbox_inches = 'tight')


#%%

patchSize = 299
xCeption = xception.Xception(include_top = False, weights = 'imagenet', pooling = 'avg', input_shape = (patchSize,patchSize,3))
outputs = xCeption.output
output1 = Dense(1, activation = 'linear', name = "mosOutput")(outputs)
output2 = Dense(2, activation = 'sigmoid', name = "typeOutput")(outputs)
model = Model(inputs = xCeption.input, outputs = [output1, output2])
model.summary()
losses = {"mosOutput": 'mean_squared_error', "typeOutput": 'binary_crossentropy'}
model.compile(optimizer='adam', loss=losses)

xTrain = np.random.random((4,299,299,3))
yTrain = [np.random.random(4), np.random.random((4,2))]

model.fit(xTrain, yTrain, epochs = 10)
model.predict(xTrain)
#%%

model = load_model("Results\\xception38every53thNew_best.model")
#model.summary()
#%%

data = pd.read_excel("J:\\DataSet\\NewDataset\\ParsedVMAF.XLSX")
folderName = "PU_30fps_30sec_Part1_1920x1080_4000_x264"
filepath = "J:\\DataSet\\NewDataset\\All\\"+folderName+"\\"

cutsx = (np.arange(np.ceil(1920/299))*299).astype(int)
cutsy = (np.arange(np.ceil(1080/299))*299).astype(int)


#%%

c = 0
patches = []
y = []
for fileIndex in np.arange(0,900,23):
    img = imageio.imread(filepath + folderName +"-"+str(fileIndex+1)+".png")
    
    temp = [[img[cutsy[i]:cutsy[i+1],cutsx[j]:cutsx[j+1],:] for j in range(len(cutsx)-1)] for i in range(len(cutsy)-1)]
    temp = np.array(temp).reshape(-1, 299, 299, 3)
    patches.extend(temp)
    y.append(data[folderName + ".yuv.txt"][fileIndex])
    c += 1

patches = xception.preprocess_input(np.array(patches)) 
preds = model.predict(patches).reshape(c, 3, 6)
errors = np.sum(np.abs(preds - np.array(y)[:,None,None])/c, axis = 0)

#%%

overlay = img[:3*299,:6*299]/20
overlay -= np.repeat(np.repeat(errors, 299, axis = 0), 299, axis = 1)[:,:,np.newaxis] - np.min(errors)
plt.figure(figsize=(14,8))
plt.imshow(overlay/np.max(overlay))
#plt.colorbar()

#%%
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))
dsTest = DataSequenceNew(filepath = "J:\\DataSet\\NewDataset\\All\\", indexes = testIndexes, patchSize = 299, everynth = 23, batchSize = 100, preprocess = xception.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1)
yTest = []
for i in range(dsTest.__len__()):
    yTest.append(dsTest.__getitem__(i))
yTest = np.concatenate(yTest)
dsTest.switchTestReturnMode()

pr = model.predict_generator(dsTest)[:,0]
np.sqrt(np.mean(np.square(pr-yTest)))
#%%

model = load_model("Results\\xception38every53thNew_best.model")
weights = model.get_weights()
model2.summary()
#%%

xCeption = xception.Xception(include_top = False, weights = 'imagenet', pooling = 'avg', input_shape = (3*299,6*299,3))
outputs = xCeption.output
outputs = Dense(1, activation = 'linear')(outputs)
model2 = Model(inputs = xCeption.input, outputs = outputs)
weights2 = model2.set_weights(weights)

data = pd.read_excel("J:\\DataSet\\NewDataset\\ParsedVMAF.XLSX")
folderName = "CSGO_30fps_30sec_Part1_1920x1080_4000_x264"
filepath = "J:\\DataSet\\NewDataset\\All\\"+folderName+"\\"

#%%
ind = np.argsort(weights[234].flatten())[0]

img = xception.preprocess_input(imageio.imread(filepath + folderName +"-"+str(0+1)+".png")[:3*299,:6*299,:])
patches = [[img[cutsy[i]:cutsy[i+1],cutsx[j]:cutsx[j+1],:] for j in range(len(cutsx)-1)] for i in range(len(cutsy)-1)]
patches = np.array(patches).reshape(-1, 299, 299, 3)
#patches = patches

(model.predict(patches))

img_pp = np.zeros((3*299, 6*299, 3), dtype = np.uint8)
for i in range(len(patches)):
    img_pp[(i//6)*299:(i//6+1)*299,(i%6)*299:(i%6+1)*299,:] = patches[i]

model2.predict(img[None])
#%% ################################ COPY every nth #############################################

import os
import shutil
src = "J:\\DataSet\\NewDataset\\All\\"
dest = "D:\\All\\"
src_folders = os.listdir(src)
for i in trainIndexes1:
    src_files = os.listdir(os.path.join(src, src_folders[i]))
    for j in range(0,900,53):
        full_file_name = os.path.join(os.path.join(src, src_folders[i]), src_folders[i] + "-" + str(j+1)+".png")
        if not os.path.exists(dest + src_folders[i]):
            os.makedirs(dest + src_folders[i])
        shutil.copy(full_file_name, dest + src_folders[i])

#%%

folderName = "Dota2_30FPS_30Sec_Part1_1920x1080_4000_x264"
filepath = "D:\\All\\"+folderName+"\\"

#%%
img = imageio.imread(filepath + folderName +"-"+str(0+1)+".png")

patches = np.zeros((1080,1920,3))

patches[:299,:299] = 102
patches[-299:,:299]  = 102
patches[:299,-299:]  = 102
patches[-299:,-299:] = 102

#patches[540-149:540+150,:299] = 102
#patches[540-149:540+150,-299:] = 102

#patches[:299,960-149:960+150]  = 102
#patches[-299:,960-149:960+150] = 102

patches[540-149:540+150,960-149:960+150] = 102
patches[195:195+299,555-149:555+150] = 102
patches[195:195+299,-555-149:-555+150] = 102
patches[-195-299:-195,555-149:555+150] = 102
patches[-195-299:-195,-555-149:-555+150] = 102

fig = plt.figure()
fig.set_size_inches((1920*0.003,1080*0.003))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('hot')
ax.imshow((img/5*3+patches).astype(int), aspect='equal')
#plt.savefig(outputname, dpi=dpi)
plt.savefig("patterns/pattern09", bbox_inches=0)
#299 512 299 511 299 | 1920
#299 +512 +299 +511 +299

#%%

img = imageio.imread(filepath + folderName +"-"+str(0+1)+".png")

patches = np.zeros((1080,1920,3))

patches[:299,:299]   = 102
patches[-299:,:299]  = 102
patches[:299,-299:]  = 102
patches[-299:,-299:] = 102

patches[540-149:540+150,:299] = 102
patches[540-149:540+150,-299:] = 102

#patches[540-149-299:540-149,960-149:960+150]  = 102
#patches[540+150:540+150+299,960-149:960+150] = 102

patches[540-149:540+150,960-149:960+150] = 102

#patches[540-299:540,960-149-299:960-149] = 102
#patches[540-299:540,960+150:960+150+299] = 102
#patches[540:540+299,960-149-299:960-149] = 102
#patches[540:540+299,960+150:960+150+299] = 102

plt.figure(figsize= (18,10))
plt.imshow((img/5*3+patches).astype(int))
plt.savefig("pattern13_1")
#299 512 299 511 299 | 1920
#299 +512 +299 +511 +299

#%%

img = imageio.imread(filepath + folderName +"-"+str(0+1)+".png")

patches = np.zeros((1080,1920,3))

patches[:299,:299]   = 102
patches[-299:,:299]  = 102
patches[:299,-299:]  = 102
patches[-299:,-299:] = 102



patches[540-299:540,960-299:960] = 102
patches[540-299:540,960:960+299] = 102

patches[540:540+299,960-299:960] = 102
patches[540:540+299,960:960+299] = 102

patches[540-150:540+149,-380-149:-380+150] = 102
patches[540-150:540+149,380-149:380+150] = 102

plt.figure(figsize= (18,10))
plt.imshow((img/5*3+patches).astype(int))
plt.savefig("pattern10")
#299 512 299 511 299 | 1920
#299 +512 +299 +511 +299

#%%

img = imageio.imread(filepath + folderName +"-"+str(0+1)+".png")

patches = np.zeros((1080,1920,3))

patches[:299,:299]   = 102
patches[-299:,:299]  = 102
patches[:299,-299:]  = 102
patches[-299:,-299:] = 102


patches[540-149:540+150,100:100+299] = 102
patches[540-149:540+150,-299-100:-100] = 102

patches[540-299:540,960-149:960+150]  = 102
patches[540:540+299,960-149:960+150] = 102

patches[540-299:540,960-149-299:960-149] = 102
patches[540-299:540,960+150:960+150+299] = 102
patches[540:540+299,960-149-299:960-149] = 102
patches[540:540+299,960+150:960+150+299] = 102

plt.figure(figsize= (18,10))
plt.imshow((img/5*3+patches).astype(int))
plt.savefig("pattern12_1")
#299 512 299 511 299 | 1920
#299 +512 +299 +511 +299


#%%

n = 7
patches = np.zeros((1080,1920,3))

for patch in PatchPatterns.patchesPerNumber[7]:
#    x, y = np.random.randint(0,1080-299), np.random.randint(0,1920-299)
    patches[patch] = 102



fig = plt.figure()
fig.set_size_inches((1920*0.003,1080*0.003))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('hot')
ax.imshow((img/5*3+patches).astype(int), aspect='equal')
#plt.savefig(outputname, dpi=dpi)
plt.savefig("patterns/for paper/pattern_random05", bbox_inches=0)



#%% ############################# k every nth  vs  k/n any ########################################

filepath = "D:\\All\\"


trainIndexes = np.arange(390)
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))
trainIndexes = np.delete(trainIndexes, testIndexes)

nTrainVideos = len(trainIndexes)
nTestVideos = len(testIndexes)

model = load_model("Results\\denseNet\\densenet204\\densenet204every53thNew_best.model")

#%%

metricsList = [[],[]]
nPatches = 13

dsRand = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = 2, batchSize = 20, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1, patchMode='random')  
yTest = []
for i in range(dsRand.__len__()):
    yTest.append(dsRand.__getitem__(i))
yTest = np.concatenate(yTest)
yTest = yTest[::1]
dsRand.switchTestReturnMode()

predRand = model.predict_generator(dsRand, verbose=1)
predRand = np.mean(predRand.reshape((-1,1)), axis = -1)

metricsList[0].append(getMetrics(predRand, yTest))
metricsList[1].append(getMetrics(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1), np.mean(predRand.reshape((nTestVideos,-1)), axis=-1), verbose=True))


dsRand2 = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = 23, batchSize = 20, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = nPatches, patchMode='random')  
yTest = []
for i in range(dsRand2.__len__()):
    yTest.append(dsRand2.__getitem__(i))
yTest = np.concatenate(yTest)
yTest = yTest[::nPatches]
dsRand2.switchTestReturnMode()
predPatt = model.predict_generator(dsRand2, verbose=1)

predPatt = np.mean(predPatt.reshape((-1,nPatches)), axis = -1)
metricsList[0].append(getMetrics(predPatt, yTest))
metricsList[1].append(getMetrics(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1), np.mean(predPatt.reshape((nTestVideos,-1)), axis=-1), verbose=True))




dsPatt = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = 23, batchSize = 20, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = nPatches, patchMode='pattern')  
yTest = []
for i in range(dsPatt.__len__()):
    yTest.append(dsPatt.__getitem__(i))
yTest = np.concatenate(yTest)
yTest = yTest[::nPatches]
dsPatt.switchTestReturnMode()
predPatt = model.predict_generator(dsPatt, verbose=1)

predPatt = np.mean(predPatt.reshape((-1,nPatches)), axis = -1)
metricsList[0].append(getMetrics(predPatt, yTest))
metricsList[1].append(getMetrics(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1), np.mean(predPatt.reshape((nTestVideos,-1)), axis=-1), verbose=True))

metricsList = np.array(metricsList)

#%%

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize = (7,7))
plt.bar(np.arange(3)*3+1, metricsList[0,:,1], align='center', alpha=0.9, width = 1, color = 'c')
plt.bar(np.arange(3)*3, metricsList[1,:,1], align='center', alpha=0.9, width = 1, color = 'g')
plt.xticks(np.arange(3)*3+0.5, ["one random crop\nevery 2nd image", "13 random crops\nevery 23th image", "13 selected crops\nevery 23th image"])
plt.ylabel('RMSE')
#plt.xlabel('Number of frames in the training set')
plt.legend(["Frame level", "Video level"], loc=1)
plt.ylim(0,7.2)

plt.savefig('Results\\comparePatternRandom13.png', bbox_inches = 'tight')


#%% #########################################    pattern   vs   random     #############################################

metricsRand = [[],[]]
metricsPatt = [[],[]]

for nPatches in [1,5,7,9,11,13]:
    
    dsPatt = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = 53, batchSize = 10, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = nPatches, patchMode='pattern1')  
    yTest = []
    for i in range(dsPatt.__len__()):
        yTest.append(dsPatt.__getitem__(i))
    yTest = np.concatenate(yTest)
    yTest = yTest[::nPatches]
    dsPatt.switchTestReturnMode()
    predPatt = model.predict_generator(dsPatt, verbose=1)
    
    predPatt = np.mean(predPatt.reshape((-1,nPatches)), axis = -1)
    metricsPatt[0].append(getMetrics(predPatt, yTest))
    metricsPatt[1].append(getMetrics(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1), np.mean(predPatt.reshape((nTestVideos,-1)), axis=-1), verbose=True))

    
    dsRand = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = 53, batchSize = 10, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 0, nPatches = nPatches, patchMode='random')  
    
    predRand = model.predict_generator(dsRand, verbose=1)
    predRand = np.mean(predRand.reshape((-1,nPatches)), axis = -1)
    
    metricsRand[0].append(getMetrics(predRand, yTest))
    metricsRand[1].append(getMetrics(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1), np.mean(predRand.reshape((nTestVideos,-1)), axis=-1), verbose=True))
#%%

metricsRand = np.array(metricsRand)
metricsPatt = np.array(metricsPatt)
#%%

from matplotlib.patches import Patch
from matplotlib.lines import Line2D



legend_elements = [Line2D([0], [0], color='r', label='random'),
                   Line2D([0], [0], color='b', label='pattern'),
                   Line2D([0], [0], marker='o', color='k', label='frame level'),
                   Line2D([0], [0], marker='x', color='k', label='video level')]

fig, ax = plt.subplots()

plt.plot([1,5,7,9,11,13], metricsRand[0,:,1], 'ro-')
plt.plot([1,5,7,9,11,13], metricsRand[1,:,1], 'rx-')
plt.plot([1,5,7,9,11,13], metricsPatt[0,:,1], 'bo-')
plt.plot([1,5,7,9,11,13], metricsPatt[1,:,1], 'bx-')
plt.ylabel("RMSE")
plt.xlabel("Number of patches")
ax.legend(handles=legend_elements, ncol=2)


#%% ######################################### CNN Visualization ######################################


model = load_model("Results\\denseNet\\densenet204\\densenet204every53thNew_best.model")
model.summary()
denseNet = densenet.DenseNet121(include_top = False, weights = None, pooling = None, input_shape = (1080,1920,3))
denseNet.set_weights(model.get_weights()[:-2])

#denseNet.summary()
yData = pd.read_excel("J:\\DataSet\\NewDataset\\ParsedVMAF.XLSX")

#%%

folderName = "FIFA17_30fps_30sec_Part2_1920x1080_2000_x264"
filepath = "J:\\DataSet\\NewDataset\\All\\"+folderName+"\\"

img_plot = np.array(imageio.imread(filepath + folderName +"-"+str(1)+".png"))
img = densenet.preprocess_input(img_plot)
#plt.imshow(img)
conv = denseNet.predict(np.expand_dims(img, axis=0))[0]

weights = model.get_weights()[-2:]


test = conv*weights[0][:,0]


pred = np.sum(conv*weights[0][:,0], axis = -1)+weights[1]
#pred = (pred - np.min(pred))
#pred = pred / np.max(pred)
#model.predict(np.expand_dims(img[:299,:299,:], axis=0))

yValue = yData[folderName +".yuv.txt"][1-1]
yPred = np.mean(pred)



#
#fig, (ax1, ax2) = plt.subplots(2, gridspec_kw = {'height_ratios':[1, 1]})
#fig.set_size_inches((10,10))
#ax1.imshow(img_plot)
#cm = ax2.matshow(pred, cmap = 'jet', vmin=-10, vmax=129) #, shading = 'gouraud'
#ax2.set_aspect(1)
#cbar = plt.colorbar(mappable = cm)
#cbar.set_ticks([0,20,40,60,80,100,120])
#cbar.set_ticklabels([0,20,40,60,80,100,120])
#cbar.ax.minorticks_on()
#
#cbar.ax.plot([-10,130], [yValue]*2, 'k')
#cbar.ax.plot([-10,130], [yPred]*2, 'w') 
#leg = cbar.ax.legend(["VMAF", "Prediction"], bbox_to_anchor=(12, .6))
#leg.get_frame().set_facecolor('gray')
#
#
#for ax in (ax1, ax2):
#    ax.set_anchor('W')
#    ax.set_yticklabels([])
#    ax.set_xticklabels([])


#%%

import matplotlib as mpl
def rgb2gray(rgb):
    return np.repeat(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])[...,None],3,2)
    
plt.rcParams.update({'font.size': 14})

cm = plt.get_cmap('jet')
pic_weight = 0.4

pred_overlay = cv2.resize(pred, dsize=(1920,1080), interpolation=cv2.INTER_CUBIC)
overlay = rgb2gray(img_plot/255.)*pic_weight

#pred_overlay = (pred_overlay - np.min(pred_overlay))
#pred_overlay = pred_overlay / np.max(pred_overlay)
colored_pred_overlay = cm(pred_overlay/120)[:,:,:3]
#plt.imshow(colored_pred_overlay)
overlay += colored_pred_overlay*(1-pic_weight)

fig = plt.figure(figsize=(11,6))
ax1 = fig.add_axes([0.00, 0.0, 0.82, 1.0])
ax2 = fig.add_axes([0.895, 0.077, 0.025, 0.72])

ax1.imshow(overlay)
ax1.axis('off')

norm = mpl.colors.Normalize(vmin=-10, vmax=130)

cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cm,
                                norm=norm)

cbar.set_ticks(np.array([0,20,40,60,80,100,120]))
cbar.ax.minorticks_on()

cbar.ax.plot([-10,130], [yValue]*2, 'k')
cbar.ax.plot([-10,130], [yPred]*2, 'w') 
leg = cbar.ax.legend(["VMAF", "Prediction"], bbox_to_anchor=(4.49, 1.2))
leg.get_frame().set_facecolor('#BBBBBB')
fig.tight_layout()
#plt.savefig("localPresLol.pdf")



#%% ############################################ Compare new and old dataset ##########################################################

filepath1 = "D:\\All\\"
filepath2 = "J:\\DeepDS\\All\\"


trainIndexes1 = np.arange(390)
testIndexes1 = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))
trainIndexes1 = np.delete(trainIndexes1, testIndexes1)

nTestVideos1 = len(testIndexes1)

testIndexes2 = np.arange(600)

nTestVideos2 = len(testIndexes2)

model = load_model("Results\\denseNet\\densenet204\\densenet204every53thNew_best.model")

#%%


ds1 = DataSequenceNew(filepath1, testIndexes1, patchSize = 299, everynth = 83, batchSize = 20, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 4, patchMode='random')  
yTest1 = []
for i in range(ds1.__len__()):
    yTest1.append(ds1.__getitem__(i))
yTest1 = np.concatenate(yTest1)
ds1.switchTestReturnMode()

ds2 = DataSequenceNew2(filepath2, testIndexes2, patchSize = 299, everynth = 2*83, batchSize = 20, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 4, patchMode='random')  
yTest2 = []
for i in range(ds2.__len__()):
    yTest2.append(ds2.__getitem__(i))
yTest2 = np.concatenate(yTest2)
ds2.switchTestReturnMode()

test2 = ds2.data

predRand = model.predict_generator(ds1, verbose=1)
predRand2 = model.predict_generator(ds2, verbose=1)

#%%

plt.rcParams.update({'font.size': 12})
plt.figure(figsize = (7,8))
plt.plot(yTest1 , predRand[:,0], 'co', markersize = .8) 
plt.plot(np.mean(yTest1.reshape((nTestVideos1,-1)), axis=-1),np.mean(predRand.reshape((nTestVideos1,-1)), axis=-1), 'go', markersize = 4)
plt.plot(np.arange(101), 'k')
plt.xlim((-10,110))
plt.ylim((-10,110))
#plt.title("Epoch %i" % (i))
plt.xlabel("Real VMAF")
plt.ylabel("Predicted VMAF")
plt.legend(["Frame level", "Video level"], loc = 2)
plt.text(0, -35, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(yTest1 , predRand[:,0]))
plt.text(0, -42, "Video Level: R²: %1.2f   RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(np.mean(yTest1.reshape((nTestVideos1,-1)), axis=-1),np.mean(predRand.reshape((nTestVideos1,-1)), axis=-1)))
plt.subplots_adjust(bottom=.2)

#%%

indexes = ["1920" in name and "Lol" in name for name in test2.columns]

plt.rcParams.update({'font.size': 12})
plt.figure(figsize = (7,8))
plt.plot(yTest2 , predRand2[:,0], 'co', markersize = .8) 
for i in range(600):
    c = "r" if indexes[i] else "g"
    plt.plot(np.mean(yTest2.reshape((nTestVideos2,-1)), axis=-1)[i:i+1],np.mean(predRand2.reshape((nTestVideos2,-1)), axis=-1)[i:i+1], c+'o', markersize = 4)
plt.plot(np.arange(101), 'k')
plt.xlim((-10,110))
plt.ylim((-10,110))
#plt.title("Epoch %i" % (i))
plt.xlabel("Real VMAF")
plt.ylabel("Predicted VMAF")
plt.legend(["Frame level", "Video level"], loc = 2)
plt.text(0, -35, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(yTest2 , predRand2[:,0]))
plt.text(0, -42, "Video Level: R²: %1.2f   RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(np.mean(yTest2.reshape((nTestVideos2,-1)), axis=-1),np.mean(predRand2.reshape((nTestVideos2,-1)), axis=-1)))
plt.subplots_adjust(bottom=.2)


np.sum(test2.values==0.0, axis=0).shape


#%%

model = load_model("Results\\denseNet\\densenet204\\densenet204every53thNew_best.model")

#%%


ds1 = DataSequenceNew(filepath1, testIndexes1, patchSize = 299, everynth = 83, batchSize = 16, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 1, nPatches = 1, patchMode='random')  


setTrainability(model, 300)
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
startTime = time.time()
model.fit_generator(ds1, epochs = 5)
timeAll = time.time() - startTime


setTrainability(model, 1)
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
startTime = time.time()
model.fit_generator(ds1, epochs = 5)
timeLast = time.time() - startTime






#%% ########################################     everynth      ####################################################


filepath = "J:\\DataSet\\NewDataset\\All\\"


trainIndexes = np.arange(390)
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))


trainIndexes = np.delete(trainIndexes, testIndexes)

nTrainVideos = len(trainIndexes)
nTestVideos = len(testIndexes)

# Test: CSGO/2, Hearthstone/2, Dota, ProjectCar # CSGO, Hearthstone, LoL, Fifa, HeroesOfTheStorm, PU

batchSize = 64
everynthTest = 23

dsTest = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = everynthTest, batchSize = 100, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1)  
yTest = []
for i in range(dsTest.__len__()):
    yTest.append(dsTest.__getitem__(i))
yTest = np.concatenate(yTest)

#%%

everynthList = [13, 27, 53, 103, 203, 403]
nSamplesList = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]

metricsListFrame = [np.zeros((i[1]-i[0],4)) for i in nSamplesList]
metricsListFrameAvg = np.zeros((len(everynthList),4))   
metricsListFrameStdError = np.zeros((len(everynthList),4))   


for sampleIndex, (i0, i1) in enumerate(nSamplesList):
    for j in range(i0, i1):
        metricsListFrame[sampleIndex][j-i0] = getMetrics(np.load("Results\\final_results\\everynth\\nPatches3\\denseNet_every%ithbs64_bestEpoch%i.npy" % (everynthList[sampleIndex], j)), yTest, verbose = False)
 

for i, metric in enumerate(metricsListFrame):
    metricsListFrameAvg[i] = np.mean(metric, axis=0)    
    metricsListFrameStdError[i] = np.std(metric, ddof=1, axis=0)/np.sqrt(len(metric))    

#%%

plt.errorbar(everynthList, metricsListFrameAvg[:,1], yerr=metricsListFrameStdError[:,1])
plt.ylim(4,8)
#plt.xscale('log')
np.save("3,7,13,27,53,103,203,403_avg", metricsListFrameAvg)
np.save("3,7,13,27,53,103,203,403_stdErr", metricsListFrameStdError)

#%% ######################################## Architectures ####################################################


filepath = "J:\\DataSet\\NewDataset\\All\\"


trainIndexes = np.arange(390)
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))


trainIndexes = np.delete(trainIndexes, testIndexes)

nTrainVideos = len(trainIndexes)
nTestVideos = len(testIndexes)

# Test: CSGO/2, Hearthstone/2, Dota, ProjectCar # CSGO, Hearthstone, LoL, Fifa, HeroesOfTheStorm, PU

batchSize = 64
everynthTrain = 53
everynthTest = 23

dsTest = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = everynthTest, batchSize = 100, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1)  
yTest = []
for i in range(dsTest.__len__()):
    yTest.append(dsTest.__getitem__(i))
yTest = np.concatenate(yTest)

#%%

modelNames = ["xception", "resnet", "densenet", "mobilenet"]
percList = ["25","50","75"]

metricsListFrame = np.zeros((len(percList),len(modelNames),4))

for i in range(len(modelNames)):
    for j in range(len(percList)):
        metricsListFrame[j,i] = getMetrics(np.load("Results\\final_results\\architectures\\comp%s_%s.npy" % (percList[j], modelNames[i])), yTest, verbose = False)

metricsListFrame = np.array(metricsListFrame)

#%% ######################################## Architectures 2 ####################################################


filepath = "J:\\DataSet\\NewDataset\\All\\"


trainIndexes = np.arange(390)
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))


trainIndexes = np.delete(trainIndexes, testIndexes)

nTrainVideos = len(trainIndexes)
nTestVideos = len(testIndexes)

# Test: CSGO/2, Hearthstone/2, Dota, ProjectCar # CSGO, Hearthstone, LoL, Fifa, HeroesOfTheStorm, PU

batchSize = 64
everynthTrain = 53
everynthTest = 23

dsTest = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = everynthTest, batchSize = 100, preprocess = densenet.preprocess_input, shuffle = False, returnMode = 2, nPatches = 1)  
yTest = []
for i in range(dsTest.__len__()):
    yTest.append(dsTest.__getitem__(i))
yTest = np.concatenate(yTest)

#%%

modelNames = ["xceptioncomp_epoch51", "resnetcomp_epoch41", "densenetcomp_epoch49"]#, "mobilenet"]

metricsListFrame = np.zeros((len(modelNames),4))

for i in range(len(modelNames)):
    metricsListFrame[i] = getMetrics(np.load("Results\\final_results\\architectures\\%s.npy" % (modelNames[i])), yTest, verbose = False)

metricsListFrame = np.array(metricsListFrame)




#%%
filepathVal = "D:\\DataSetImage\\All\\"
filepathMOS = "D:\\DataSetImage\\All\\ImageDataset_Mos.xlsx"
modelPath = "D:\\NR\\Results\\densenet_bestModelafterMOStraining.model"
valIndexes = np.arange(124,164) 

model = load_model(modelPath)

nPatchesList = [1,5,7,9,11,13]
for nPatches in nPatchesList:
    print(nPatches)
    xVal,   yVal   = importDataSubjectiveFrames(filepathVal, filepathMOS, valIndexes, patchSize=299, nPatches=nPatches, patchMode='pattern' if nPatches > 1 else 'random', preprocess=densenet.preprocess_input)
    yPred = model.predict(xVal)[:,0]
    getMetrics(np.mean(yPred.reshape((len(valIndexes),nPatches)), axis=-1), yVal[::nPatches])
    
# 6: RMSE: 0.3391    PCC: 0.9645
