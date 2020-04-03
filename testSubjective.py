# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:14:14 2019

@author: utke.markus
"""


import numpy as np
import pandas as pd
import imageio
from matplotlib import pyplot as plt
from os import path
import glob
np.random.seed(7)

import tensorflow as tf
tf.Session()
tf.set_random_seed(9)
from keras.applications import densenet
from keras.models import Sequential, load_model

from importData import extractPatchPattern, extractRandomPatches
from tools import setTrainability, ValidationCallback, getMetrics

#%%
modelPath = "D:\\NR\\Results\\densenet_bestModelafterMOStraining.model"



### For GamingVideoSET
#filepath = "J:\\DataSet\\Test_GamingVideoDataSet"
#mosPath = "J:\\DataSet\\Test_GamingVideoDataSet\\SummaryOfAllResults.xlsx"
#sheet_name = "SubjectiveResults"
## title for saving the results
#title = "FinalResults_GAMINGSET"


## For testing set:
#filepath = "J:\\TestData\\All"
#mosPath = "J:\\TestData\\All\\KUGVD_IndividualSubjectiveScoresWithMOS.xlsx"
#sheet_name = "Sheet1"
# title for saving the results
filepath = "J:\\DataSet\\Test_GamingVideoDataSet" 
mosPath = "J:\\DataSet\\Test_GamingVideoDataSet\\SummaryOfAllResults.xlsx"
sheet_name = "ObjAndMOS"
##########


title = "FinalResults_testset"

# parameters for training and validation
patchSize = 299       # size of the quadratic patches
nPatches = 13         # number of patches for each frame for testing
everynth = 23         # distance between frames that are used for testing; only every nth image is used
patchMode = 'pattern' # either 'random' for random patches or 'pattern'

testIndexes = np.arange(90)

## Only for testing set, because some images in these folders cannot be loaded:
testIndexes = np.delete(testIndexes, 29)
testIndexes = np.delete(testIndexes, 69)
#testIndexes = np.concatenate([
#        np.arange(18, 33),
#        np.arange(122, 140),
#        np.arange(158, 176),
#        np.arange(176, 191),
#        np.arange(257, 272),
#        np.arange(277, 282)])
#%%

nTestImages = len(testIndexes)

# load data and model
yData = pd.read_excel(mosPath, sheet_name=sheet_name, index_col=0)["MOS"]
model = load_model(modelPath)

#%%

yPred = []
yTest = []
fileNames = []

for (i, folder) in enumerate(np.flip(yData.index.values[testIndexes])):
    print("\rPredicting: %i/%i" % (i+1, len(yData.index.values[testIndexes])), end="")
    X = []

    for fileIndex in range(0,900,everynth):
#        ## For GamingVideoSET
#        temp = imageio.imread(filepath + "\\" + folder[:-4] + "\\" + folder[:-4] + '-' + str(fileIndex+1) + ".png")
        ## For test set
        temp = imageio.imread(filepath + "\\" + folder + "\\" + folder + '_' + str(fileIndex+1).zfill(4) + ".png")
        if patchMode == 'pattern':
            X.append(extractPatchPattern(temp, nPatches, patchSize))
        elif patchMode == 'random':
            X.append(extractRandomPatches(temp, nPatches, patchSize))
        else:
            raise Exception("Please choose patchMode either as 'random' or 'pattern'.")
        fileNames.append(folder + '_' + str(fileIndex+1).zfill(4) + ".png")

    X = np.concatenate(X)            
    X = X.reshape((-1, patchSize, patchSize, 3))

    X = densenet.preprocess_input(X)

    pred = model.predict(X)
    yPred.append(np.mean(pred))
    yTest.append(yData[folder])
    
print()
    
met = getMetrics(yPred, yTest)

yData.index.values

plt.plot(yTest, yPred, "o")
plt.plot([0,5], [0,5])

np.savetxt("Test.csv", yTest, delimiter=",")
np.savetxt("Predict.csv", yPred, delimiter=",")

np.save(title, met)
predictionsByName = dict(zip(fileNames, yPred))

################### Reults old:
# first result on GamingVideoSET: R²: 0.8419       RMSE: 0.3568    PCC: 0.9286     SRCC: 0.9312
#                       test set: R²: 0.6844       RMSE: 0.5878    PCC: 0.9349     SRCC: 0.9312
################### Reults new?:
# first result on GamingVideoSET:
#                       test set: R²: 0.6500       RMSE: 0.6190    PCC: 0.9361     SRCC: 0.9314
