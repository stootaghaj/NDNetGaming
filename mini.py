# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:49:01 2018

@author: utke.markus
"""

import numpy as np
np.random.seed(7)
from scipy import stats
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm
import tensorflow as tf
tf.Session()
tf.set_random_seed(9)
from keras.activations import relu
from keras.layers import Dense, Conv2D, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, TerminateOnNaN
from keras import initializers
from keras.applications import imagenet_utils
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
import keras.backend as K
import matplotlib.pyplot as plt
import time
from importData import DataSequenceNew


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
        

#%%

patchSize = 299
nPatches = 1
model = Sequential()
conv1 = Conv2D(10, (5,5), activation='relu', input_shape=(patchSize,patchSize,3))
conv2 = Conv2D(15, (3,3), activation='relu')
pool = GlobalMaxPooling2D()
#model.add(Flatten())
model.add(conv1)
model.add(conv2)
model.add(pool)
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
#setTrainability(model, 204)#+5*10) 


#sgd = SGD(lr = 0.001, momentum = 0.0, decay = 0.0)
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
#print(model.summary())


filepath = "D:\\All\\"
filepathTest = "D:\\All\\"

trainIndexes = np.arange(390)
testIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282)))#np.hstack((np.arange(105, 140), np.arange(209, 239),  np.arange(282, 318)))
trainIndexes = np.delete(trainIndexes, testIndexes)

nTrainVideos = len(trainIndexes)
nTestVideos = len(testIndexes)

# Test: CSGO/2, Hearthstone/2, Dota, ProjectCar # CSGO, Hearthstone, LoL, Fifa, HeroesOfTheStorm, PU

batchSize = 64
everynthTrain = 53
everynthTest = 23

#%%
#class LossHistory(Callback):
#    def on_train_begin(self, logs={}):
#        self.losses = []
#
#    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))


def preprocess_input(X):
    return X/255.


class validationCallback(Callback): #TODO pass test indexes, nPatches
    def on_train_begin(self, logs={}):
        self.testIndexes = testIndexes
        self.nPatches = nPatches
        self.metricsPerFrame = []
        self.metricsPerVideo = []
        self.preds = []
        self.dsTest = DataSequenceNew(filepathTest, self.testIndexes, patchSize = patchSize, everynth = everynthTest, batchSize = 10, preprocess = preprocess_input, shuffle = False, returnMode = 2, nPatches = self.nPatches)  
        yTest = []
        for i in range(self.dsTest.__len__()):
            yTest.append(self.dsTest.__getitem__(i))
        yTest = np.concatenate(yTest)
        self.yTestPerPic = yTest[::self.nPatches]
        self.dsTest.switchTestReturnMode()
        
    def on_epoch_end(self, epoch, logs={}):
        predTest = self.model.predict_generator(self.dsTest) 
        predTestPerPic = np.mean(predTest.reshape((-1,self.nPatches)), axis = -1)
        self.preds.append(predTestPerPic)
        self.metricsPerFrame.append(getMetrics(predTestPerPic, self.yTestPerPic, verbose=True))
        self.metricsPerVideo.append(getMetrics(np.mean(predTestPerPic.reshape((len(self.testIndexes),-1)), axis=-1),np.mean(self.yTestPerPic.reshape((len(self.testIndexes),-1)), axis=-1), verbose=True))
        if np.argmin(np.array(self.metricsPerFrame)[:,1]) == epoch:
            print("New Best Epoch: %i" % epoch)
            self.model.save("Results\denseNet204every53thNewTest_best.model")
            self.bestEpoch = epoch


#%%

validationLossHistory = validationCallback()
#lossHistory = LossHistory()

dsa = DataSequenceNew(filepath, trainIndexes, patchSize = patchSize, nPatches = nPatches,
                      everynth = everynthTrain, batchSize = 16, preprocess = preprocess_input)
time1 = time.time()
model.fit_generator(dsa, steps_per_epoch = None, epochs = 5, verbose = 1, callbacks=[validationLossHistory])#, lossHistory, TerminateOnNaN()])
time2 = time.time() - time1



#%%
#plt.figure()
#plt.plot(lossHistory.losses)

plt.rcParams.update({'font.size': 12})
yTest = validationLossHistory.yTestPerPic
plt.figure(figsize = (7,8))
i = 2
plt.plot(yTest , validationLossHistory.preds[i], 'co', markersize = .8) 
plt.plot(np.mean(yTest.reshape((nTestVideos,-1)), axis=-1),np.mean(validationLossHistory.preds[i].reshape((nTestVideos,-1)), axis=-1), 'go', markersize = 4)
plt.plot(np.arange(101), 'k')
plt.xlim((-10,110))
plt.ylim((-10,110))
#plt.title("Epoch %i" % (i))
plt.xlabel("Real VMAF")
plt.ylabel("Predicted VMAF")
plt.legend(["Frame level", "Video level"], loc = 2)
plt.text(0, -35, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % validationLossHistory.metricsPerFrame[i])
plt.text(0, -42, "Video Level: R²: %1.2f   RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % validationLossHistory.metricsPerVideo[i])
plt.subplots_adjust(bottom=.2)
#plt.savefig("Results\\denseNet204every53thNewTest_epoch%i.png" % (i), bbox_inches = 'tight')
#plt.clf()
#%%
#np.save("Results\\denseNet204every53thNewTest_epoch%i", validationLossHistory.preds[i])
#%%

weights = model.get_weights()
kernels = weights[0]
#kernels = weights[2]

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

for (i, ax) in enumerate(axes.flat):
    ax.set_axis_off()
    im = ax.imshow(kernels[:,:,0,i], cmap='bwr_r',
                   vmin=np.min(kernels), vmax=np.max(kernels))

# notice that here we use ax param of figure.colorbar method instead of

# the cax param as the above example

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)


plt.show()


#%%

modelPart = Sequential()
#conv1 = Conv2D(10, (5,5), activation='relu', input_shape=(patchSize,patchSize,3))
#conv2 = Conv2D(15, (3,3), activation='relu')
#pool = GlobalMaxPooling2D()
#model.add(Flatten())
modelPart.add(conv1)
modelPart.add(conv2)
modelPart.add(pool)
modelPart.summary()

#%%

dsPart = DataSequenceNew(filepath, trainIndexes, patchSize = 299, everynth = 53, batchSize = 64, preprocess = preprocess_input, shuffle = False, returnMode = 2, nPatches = nPatches, patchMode='random')  
yPart = []
for i in range(dsPart.__len__()):
    yPart.append(dsPart.__getitem__(i))
yPart = np.concatenate(yPart)
yPart = yPart[::nPatches]
dsPart.switchTestReturnMode()

xPart = modelPart.predict_generator(dsPart, verbose=1)

#%%

dsPartTest = DataSequenceNew(filepath, testIndexes, patchSize = 299, everynth = 53, batchSize = 64, preprocess = preprocess_input, shuffle = False, returnMode = 2, nPatches = nPatches, patchMode='random')  
yPartTest = []
for i in range(dsPartTest.__len__()):
    yPartTest.append(dsPartTest.__getitem__(i))
yPartTest = np.concatenate(yPartTest)
yPartTest = yPartTest[::nPatches]
dsPartTest.switchTestReturnMode()

xPartTest = modelPart.predict_generator(dsPartTest, verbose=1)

#%%
from sklearn.ensemble import RandomForestRegressor

svr = RandomForestRegressor(n_estimators=100, max_depth=None, max_features='auto', n_jobs=4, verbose=1)

svr.fit(xPart, yPart)

yPartTestPred = svr.predict(xPartTest)

yPartTrainPred = svr.predict(xPart)
#%%


plt.rcParams.update({'font.size': 12})
plt.figure(figsize = (7,8))
i = 2
plt.plot(yPartTest, yPartTestPred, 'co', markersize = .8) 
plt.plot(np.mean(yPartTest.reshape((nTestVideos,-1)), axis=-1),np.mean(yPartTestPred.reshape((nTestVideos,-1)), axis=-1), 'go', markersize = 4)
plt.plot(np.arange(101), 'k')
plt.xlim((-10,110))
plt.ylim((-10,110))
#plt.title("Epoch %i" % (i))
plt.xlabel("Real VMAF")
plt.ylabel("Predicted VMAF")
plt.legend(["Frame level", "Video level"], loc = 2)
plt.text(0, -35, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(yPartTestPred, yPartTest))
plt.text(0, -42, "Video Level: R²: %1.2f   RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % (1,2,3,4))
plt.subplots_adjust(bottom=.2)

#%%


plt.rcParams.update({'font.size': 12})
plt.figure(figsize = (7,8))
i = 2
plt.plot(yPart, yPartTrainPred, 'co', markersize = .8) 
plt.plot(np.mean(yPart.reshape((nTrainVideos,-1)), axis=-1),np.mean(yPartTrainPred.reshape((nTrainVideos,-1)), axis=-1), 'go', markersize = 4)
plt.plot(np.arange(101), 'k')
plt.xlim((-10,110))
plt.ylim((-10,110))
#plt.title("Epoch %i" % (i))
plt.xlabel("Real VMAF")
plt.ylabel("Predicted VMAF")
plt.legend(["Frame level", "Video level"], loc = 2)
plt.text(0, -35, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(yPartTrainPred, yPart))
plt.text(0, -42, "Video Level: R²: %1.2f   RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % (1,2,3,4))
plt.subplots_adjust(bottom=.2)