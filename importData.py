# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:32:28 2018

@author: utke.markus
"""


import glob
from os import path
from pathlib import Path
import numpy as np
np.random.seed(7)
import imageio
import pandas as pd
from sklearn.feature_extraction import image
import tensorflow as tf
tf.set_random_seed(9)
from keras.utils import to_categorical, Sequence
from tools import extractRandomPatches, extractPatchPattern

#%%


def importLIVEData(nPatchesTrain, nPatchesVal, patchSize):
    nImages = 233
    data_folder = Path("data\\live2013")
    df_labels1 = pd.read_csv(data_folder / "subjectscores1.txt", sep = ' ', header = None)
    df_labels2 = pd.read_csv(data_folder / "subjectscores2.txt", sep = ' ', header = None)
    
    df_labels = pd.concat([df_labels1, df_labels2], ignore_index = True)
    df_labels_mean = df_labels.mean(axis = 1)
    
    df_info = pd.read_csv(data_folder / "jpeginfo.txt", sep = ' ', header = None)
    origins = df_info.iloc[:,0].unique()
    valBools = df_info.loc[:,0].isin(origins[:10])
    
    trainIndices = np.arange(nImages)[~valBools]
    valIndices = np.arange(nImages)[valBools]
    
    nTrain = len(trainIndices)
    nVal = len(valIndices)
    
    xTrain = np.zeros((nTrain*nPatchesTrain, patchSize, patchSize, 3))
    xVal = np.zeros((nVal*nPatchesVal, patchSize, patchSize, 3))
    
    for i in range(nTrain):
        temp = imageio.imread(data_folder / ("img"+str(trainIndices[i]+1)+".bmp"))
        xTrain[i*nPatchesTrain:(i+1)*nPatchesTrain] = image.extract_patches_2d(temp, (patchSize,patchSize), max_patches=nPatchesTrain)
    
    for i in range(nVal):
        temp = imageio.imread(data_folder / ("img"+str(valIndices[i]+1)+".bmp"))
        xVal[i*nPatchesVal:(i+1)*nPatchesVal] = image.extract_patches_2d(temp, (patchSize,patchSize), max_patches=nPatchesVal)
    
    
    yTrain = np.repeat(df_labels_mean.values[trainIndices], nPatchesTrain)/100 
    yVal =  np.repeat(df_labels_mean.values[valIndices], nPatchesVal)/100 
    
    return xTrain, yTrain, xVal, yVal

#%%

def importTIDData(nPatchesTrain, nPatchesVal, patchSize):
    nImages = 24
    nVal = 8
    data_folder = Path("data\\tid2013")
        
    df_labels = pd.read_csv(data_folder / "mos.txt", sep = ' ', header = None)
    
    nTrain = nImages-nVal
    
    
    
    xTrain = np.zeros((nTrain, 24, 5, nPatchesTrain, patchSize, patchSize, 3), dtype = np.uint8)
    xVal = np.zeros((nVal, 24, 5, nPatchesVal, patchSize, patchSize, 3), dtype = np.uint8)
    
    for i in range(nTrain):
        for j in range(24):
            for k in range(5):
                temp = imageio.imread(data_folder / ("distorted_images\\i"+str(i+1).zfill(2)+'_'+str(j+1).zfill(2)+'_'+str(k+1)+".bmp"))
                xTrain[i,j,k] = image.extract_patches_2d(temp, (patchSize,patchSize), max_patches=nPatchesTrain)
    
    for i in range(nVal):
        for j in range(24):
            for k in range(5):
                temp = imageio.imread(data_folder / ("distorted_images\\i"+str(nTrain+i+1).zfill(2)+'_'+str(j+1).zfill(2)+'_'+str(k+1)+".bmp"))
                xVal[i,j,k] = image.extract_patches_2d(temp, (patchSize,patchSize), max_patches=nPatchesVal)
                
    xTrain = xTrain.reshape((nTrain*24*5*nPatchesTrain, patchSize, patchSize, 3))
    xVal = xVal.reshape((nVal*24*5*nPatchesVal, patchSize, patchSize, 3))
    
    yTrain = np.repeat(df_labels.values[:nTrain*24*5], nPatchesTrain)/9 
    yVal =  np.repeat(df_labels.values[nTrain*24*5:nImages*24*5], nPatchesVal)/9

    return xTrain, yTrain, xVal, yVal


#%%
    
class DataSequenceNew(Sequence):
    '''     Imports the data and crops the patches, can be passed to keras as a generator
    
            **Parameters**:
                * filepath:   Filepath to the folder in which are the video folders containing the frames of each video
                * indexes:    Indexes of the folders to use (the sort of ``glob.glob(filepath + "\\*")`` is used, might be different from the sort in the file explorer)
                * batchSize:  Size of the batches
                * patchSize:  Size of the quadratic patches
                * nPatches:   Number of patches to take from each frame in each epoch (careful: the batch size will be multiplicated accordingly, patches of same image will be next to each other in the batch)
                * patchMode:  Either 'random' for random patches or 'pattern' for choosing patches using a pattern
                * everynth:   Distance between frames that are used; only every nth image is used
                * shuffle:    Whether to shuffle the order of the traing set  after each epoch, usually True for training and False for validation
                * preprocess: Method that should be used to preprocess (scale) the data
                * returnMode: Either 0 to return only the training data, 2 to return only the labels or 1 to return a tuple of both
    
            **Public Methods**:
                * switchValReturnMode: switches returnMode from 0 to 2 or 2 to 0, does nothing if returnMode is 1
    '''
    def __init__(self, filepath, filepathVMAF, indexes, batchSize=8, patchSize=299, nPatches = 1, patchMode = 'random', everynth = 53, shuffle=True, preprocess = None, returnMode = 1): 
        self.patchSize = patchSize
        self.batchSize = batchSize
        self.nPatches = nPatches
        self.patchMode = patchMode
        self.filepath = filepath
        self.locations = [path.basename(folder) for folder in glob.glob(filepath + "\\*")]
        self.shuffle = shuffle
        self.returnMode = returnMode
        self.preprocess = preprocess
        self.data = pd.read_excel(filepathVMAF)

        self.indexes = np.transpose([np.repeat(indexes, np.ceil(900/everynth)), np.tile(np.arange(1,901, everynth), len(indexes))]).astype(np.int16)

        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batchSize))

    def __getitem__(self, index):
#        print("\nget item: %i, returnMode: %i" % (index,self.returnMode))

        if (index+1)*self.batchSize < len(self.indexes):
            indexes_temp = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        else:
            indexes_temp = self.indexes[index*self.batchSize:]
            
        return self.__genarateData(indexes_temp)

    def on_epoch_end(self):
        if self.shuffle and self.returnMode == 1:
            np.random.shuffle(self.indexes)
        
        
    def __genarateData(self, indexes_temp):
        if self.returnMode < 2:
            X = np.zeros((len(indexes_temp), self.nPatches, self.patchSize,self.patchSize,3))
        
            for i, (folderIndex, fileIndex) in enumerate(indexes_temp):
                temp = imageio.imread(self.filepath + "\\" + self.locations[folderIndex] + "\\" + self.locations[folderIndex] + '-' + str(fileIndex) + ".png")
                if self.patchMode == 'pattern':
                    X[i,] = extractPatchPattern(temp, self.nPatches, self.patchSize)
                elif self.patchMode == 'random':
                    X[i,] = extractRandomPatches(temp, self.nPatches, self.patchSize)
                else:
                    raise Exception("Please choose patchMode either as 'random' or 'pattern'.")
                    
            X = X.reshape((len(indexes_temp) * self.nPatches, self.patchSize, self.patchSize, 3))

            if self.preprocess != None:
                X = self.preprocess(X)

            assert not np.isnan(np.sum(X))

        if self.returnMode > 0:
            y = np.zeros((len(indexes_temp), self.nPatches))

            for i, (folderIndex, fileIndex) in enumerate(indexes_temp):
                y[i] = self.data[self.locations[folderIndex]+".yuv.txt"][fileIndex-1]
        
            y = y.reshape((len(indexes_temp) * self.nPatches))
        
            assert not np.isnan(np.sum(y))
        
        if self.returnMode == 0:
            return X
        elif self.returnMode == 1:
            return X, y
        elif self.returnMode == 2:
            return y
        
    def switchValReturnMode(self):
        self.returnMode = 2 - self.returnMode
        

        
#%%
        
def importDataSubjectiveFrames(filepath, filepathMOS, sheetNameMOS, labelType, indexes, patchSize=299, nPatches = 13, patchMode = 'pattern', preprocess = None, returnMode = 1):
    
    xData = np.zeros((len(indexes), nPatches, patchSize, patchSize, 3))
    yData = np.zeros((len(indexes), nPatches))
    imagePaths = glob.glob(filepath + "\\*.png")
    labelData = pd.read_excel(filepathMOS, sheet_name = sheetNameMOS, index_col=0).T
    for i, index in enumerate(indexes):
        img = imageio.imread(imagePaths[index])
        if(returnMode < 2):
            if (patchMode == 'pattern'):
                xData[i] = extractPatchPattern(img, nPatches, patchSize)
            else:
                xData[i] = extractRandomPatches(img, nPatches, patchSize)
        if (returnMode > 0):
            yData[i] = labelData[path.basename(imagePaths[index])][labelType]
    
    xData = xData.reshape((len(indexes)* nPatches, patchSize, patchSize, 3))
    yData = yData.reshape((len(indexes)* nPatches))
    if (preprocess):
        xData = preprocess(xData)
    if (returnMode == 0):
        return xData
    elif returnMode == 1:
        return xData, yData
    else:
        return yData


#%%
        
def importDataSubjective(filepath, indexes, patchSize=299, nPatches = 13, patchMode = 'pattern', preprocess = None, returnMode = 1):
    xData = np.zeros((len(indexes), nPatches, patchSize, patchSize, 3))
    yData = np.zeros((len(indexes), nPatches))
    imagePaths = glob.glob(filepath + "\\*.png")
    labelData = pd.read_excel(filepath + "ImageDataset_Mos.xlsx", index_col=0).T
    for i, index in enumerate(indexes):
        img = imageio.imread(imagePaths[index])
        if(returnMode < 2):
            if (patchMode == 'pattern'):
                xData[i] = extractPatchPattern(img, nPatches, patchSize)
            else:
                xData[i] = extractRandomPatches(img, nPatches, patchSize)
        if (returnMode > 0):
            yData[i] = labelData[path.basename(imagePaths[index])][0]
    
    xData = xData.reshape((len(indexes)* nPatches, patchSize, patchSize, 3))
    yData = yData.reshape((len(indexes)* nPatches))
    if (preprocess):
        xData = preprocess(xData)
    if (returnMode == 0):
        return xData
    elif returnMode == 1:
        return xData, yData
    else:
        return yData


#%%
          

        
        
        
