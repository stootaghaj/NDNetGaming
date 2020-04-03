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
from keras.applications import densenet, xception, resnet50, mobilenet_v2
from keras.models import Sequential, load_model

from importData import importDataSubjectiveFrames
from tools import plotPredictionSubjective, setTrainability, ValidationCallbackSubjective, buildModel
import pandas as pd


#%%

# filepaths for training and validation set
filepathTrain = "D:\\DataSetImage\\All\\"
filepathVal = "D:\\DataSetImage\\All\\"
filepathMOS = "D:\\DataSetImage\\TrainTest\\Train\\Copy of ImageDataset_DMOS(1).xlsx"
sheetNameMOS = "MOS_DMOS"
modelPath = "D:\\NR\\Results\\densenet_bestModelafterVMAFtraining.model"


# use labalType =  0: MOS
#                  1: Fragmentation
#                  2: Unclrearness
#                  3: VQ_DMOS
#                  4: VF_DMOS
#                  5: VU_DMOS
labelType = 6


    # title for saving the model
title = "subjectiveDemo2_DMOS_Final"

# parameters for training and validation
patchSize = 299           # size of the quadratic patches
nPatchesTrain = 13        # number of patches for each frame for training
nPatchesTest = 13         # number of patches for each frame for testing
batchSize = 16            # batch size for training
epochs = 150              # number of epochs for training
nTrainableLayers = 36     # number of trainable layers, 16 layers means 4 convolutional layers in case of densenet
patchMode = 'pattern'     # either 'random' for random patches or 'pattern'

# disjunct lists of training and validation indices
# use '[x for x in enumerate(glob.glob(filepath + "\\*.png"))]' to show the list of all file names with their indices
trainIndexes = np.arange(0,164)
valIndexes = np.arange(0,164) 


#%%

nTrainImages = len(trainIndexes)
nValImages   = len(valIndexes)

# load data
xTrain, yTrain = importDataSubjectiveFrames(filepathTrain, filepathMOS, sheetNameMOS, labelType, trainIndexes, patchSize=patchSize, nPatches=nPatchesTrain, patchMode=patchMode, preprocess=densenet.preprocess_input)
xVal,   yVal   = importDataSubjectiveFrames(filepathVal,   filepathMOS, sheetNameMOS, labelType, valIndexes,   patchSize=patchSize, nPatches=nPatchesTest, patchMode=patchMode, preprocess=densenet.preprocess_input)

#%%

# callback to track the validation loss
validationLossHistory = ValidationCallbackSubjective(xVal, yVal, nPatchesTest, title)

# build the model
model = load_model(modelPath)

# set the trainability of all the layers, except the last nTrainableLayers to false
setTrainability(model, nTrainableLayers)

# compile the model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# print model summary
#model.summary()

#%%

# train the model
model.fit(xTrain, yTrain, batch_size = batchSize, epochs = epochs, callbacks = [validationLossHistory], verbose = 1)#, lossHistory, TerminateOnNaN()])

#%%
bestEpoch = validationLossHistory.bestEpoch
plotPredictionSubjective(validationLossHistory.preds[bestEpoch], yVal[::nPatchesTest], title = title)

bestEpoch = validationLossHistory.bestEpoch
np.save(title, validationLossHistory.metricsPerFrame[bestEpoch])

agg = [validationLossHistory.preds[bestEpoch], yVal[::nPatchesTest]]
aggDF=pd.DataFrame(agg)
aggDF.to_csv('Finalsrdmos1.csv', index=False)
