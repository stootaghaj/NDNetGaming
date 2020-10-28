# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:49:01 2018

@author: utke.markus

-------------------------------            IMPORTANT NOTE:            --------------------------------------

Please use this fork of keras for good results:
https://github.com/datumbox/keras/tree/bugfix/trainable_bn

Read about why this is necessary here:
http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
"""

import numpy as np
np.random.seed(7)

import tensorflow as tf
tf.Session()
tf.set_random_seed(9)
from keras.applications import densenet, xception, resnet50, mobilenet_v2, vgg19

from importData_VGG import DataSequenceNew
from tools import plotPrediction, setTrainability, ValidationCallback, buildModel



#%%

# filepaths for training and validation set
#filepathTrain = "D:\\All\\"  
filepathTrain = "J:\\DataSet\\NewDataset\\All\\"
#filepathVal = "D:\\All\\"

filepathVal = "J:\\DataSet\\NewDataset\\All\\"
filepathVMAF = "J:\\DataSet\\NewDataset\\ParsedVMAF.XLSX"

# title for saving the model
title = "VGGSaman2"

# parameters for training and validation
patchSize = 96          # size of the quadratic patches 299
nPatches = 9              # number of patches for each frame
batchSize = 10            # batch size for training
everynthTrain = 30        # distance between frames that are used for training; only every nth image is used
everynthVal = 23          # distance between frames that are used for validation; only every nth image is used
epochs = 100               # number of epochs for training
nTrainableLayers = 204    # number of trainable layers

modelBuilder =   vgg19.VGG19 #DenseNet121 # choose: xception.Xception, resnet50.ResNet50, densenet.DenseNet121, mobilenet_v2.MobileNetV2, ...

# disjunct lists of training and validation indices
# use '[x for x in enumerate(glob.glob(filepath + "\\*"))]' to show the list of all folder names with their indices
# in this case the validation set consists of: CSGO_Part2, Hearthstone_Part2, Dota_Part1, Dota_Part2, ProjectCar_Part1, ProjectCar_Part2
trainIndexes = np.arange(390)
valIndexes = np.hstack((np.arange(18, 33), np.arange(191, 209), np.arange(69, 105),  np.arange(272, 282))) 
trainIndexes = np.delete(trainIndexes, valIndexes)


#%%

nTrainVideos = len(trainIndexes)
nValVideos   = len(valIndexes)

# generators for training and validation
dsVal = DataSequenceNew(filepathVal, filepathVMAF, valIndexes, patchSize = patchSize, everynth = everynthVal, batchSize = 10, preprocess = vgg19.preprocess_input, shuffle = False, returnMode = 2, nPatches = nPatches)
dsTrain = DataSequenceNew(filepathTrain, filepathVMAF, trainIndexes, patchSize = patchSize, nPatches = nPatches, everynth = everynthTrain, batchSize = batchSize, preprocess = vgg19.preprocess_input)

# callback to track the validation loss
validationLossHistory = ValidationCallback(dsVal, nValVideos, title)

# build the model
model = buildModel(modelBuilder, patchSize)

# set the trainability of all the layers, except the last nTrainableLayers to false
setTrainability(model, nTrainableLayers)

# compile the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# print model summary
#model.summary()

#%%

# train the model
model.fit_generator(dsTrain, steps_per_epoch = None, epochs = epochs, verbose = 1, callbacks=[validationLossHistory])#, lossHistory, TerminateOnNaN()])

#%%
#plot the predictions and 
bestEpoch = validationLossHistory.bestEpoch
plotPrediction(validationLossHistory.preds[bestEpoch], validationLossHistory.yVal, validationLossHistory.nValVideos, title)

bestEpoch = validationLossHistory.bestEpoch
np.save(title, (validationLossHistory.metricsPerFrame[bestEpoch], validationLossHistory.metricsPerVideo[bestEpoch]))
