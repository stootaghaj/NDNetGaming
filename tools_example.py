# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:20:07 2019

@author: utke.markus
"""
import pandas as pd
from keras.applications import densenet, xception, resnet50, mobilenet_v2
from keras.models import load_model
from tools import plotLocalPredictions, countLayers
import glob



#%%      ----------------------------------------    Local Prediction Plot    ---------------------------------------------------------

# change the following parameters to your liking
#modelPath = "Results\\densenet_bestModelafterMOStraining.model"
modelPath = "Results\\subjectiveDemo2_DMOS_UC.model"
model = load_model(modelPath)

#%%
#image_list = []
#for imname in glob.glob('D:\\DataSetImage\\All\\*.png'): #assuming gif
    #im=Image.open(filename)
    #image_list.append(im)
fileName = "ProjectCars_30FPS_30Sec_Part1_640x480_1200_x264-107.png"
    #fileName = "CSGO_30fps_30sec_Part2_0744.png"
    #part = imname.split("\\")
    #fileName = part[3];
imgNumber = 1
smooth = True
nPatches = 9 #can be None if patches should not be considered
    
weightImage = 9
weightHeatmap = 2
weightPatches = 3
    
filepath = "D:\\DataSetImage\\All\\"
    #data = pd.read_excel("D:\\DataSetImage\\All\\ImageDataset_Mos.xlsx", index_col=0).T
data = pd.read_excel("D:\\DataSetImage\\All\\Copy of ImageDataset_DMOS(1).xlsx", index_col=0).T
    
mos = data[fileName][7]
    
img_path = filepath+fileName
plotLocalPredictions(model, densenet.DenseNet121, densenet.preprocess_input, img_path, mos=mos, smooth=smooth, nPatches=nPatches, img_weights = (weightImage,weightHeatmap,weightPatches), filename = fileName+"_localPreds")
    
    
    
    #%%      -----------------------------------------      Count layers      -----------------------------------------------------------
    
model = densenet.DenseNet121(weights = None)
countLayers(model, 7)
