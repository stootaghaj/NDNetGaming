# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:12:16 2019

@author: OmenG
"""


import numpy as np
import imageio
np.random.seed(7)

import tensorflow as tf
tf.Session()
tf.set_random_seed(9)
from keras.models import  load_model
from keras.applications import densenet
import glob
import pandas as pd
import os

modelPath =  "D:\\NR\\Results\\subjectiveDemo2_DMOS_Final.model"    #densenet_bestModelafterMOStraining.model ## "D:\\NR\\Results\\densenet_bestModelafterMOStraining.model"  #D:\\NR\\subjectiveDemo2_DMOS_VF.model" #densenet_bestModelafterVMAFtraining.model"   #
model = load_model(modelPath)

#%%J:\TestData\All\temo J:\DataSet\Test_GamingVideoDataSet\Image H:\Download\YUV\rest J:\image_NETFLIX  \Test4\ C:\Users\OmenG\Desktop\Dataset
Path =[x[0] for x in os.walk('C:\\Users\\OmenG\\Desktop\\Dataset\\IMG\\')] 
for pathname in Path:
    image_list = []
    image_name = []
    #for filename in glob.glob('D:\\DataSetImage\\All\\test\\*.png'): #assuming png  H:\Download\netflix_live\done
    #for filename in glob.glob('D:\\NR\\data\\tid2013\\distorted_images\\*.bmp'): #assuming png
    for filename in glob.glob(pathname + '\\*.png'):
    
        im=np.array(imageio.imread(filename)) 
        image_list.append(im)
        image_name.append(filename)
    #MOS = np.empty(np.size(image_list,0))
    MOS = np.empty(len(image_list))
    count = 0;
    #imagePath = CSGO_30fps_30sec_Part1_640x480_600_x264-743.png"
    for ims in image_list:
        
        patches = np.zeros((ims.shape[0]//299, ims.shape[1]//299, 299, 299, 3))
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patches[i,j] = ims[i*299:(i+1)*299, j*299:(j+1)*299]
                
                
        patches = densenet.preprocess_input(patches.reshape((-1, 299, 299, 3)))
        preds = model.predict(patches)
        avgPred = np.mean(preds)
        MOS[count]= avgPred;
        del avgPred, preds, patches;
        count = count + 1;
    print(MOS)
    
    df = pd.DataFrame (MOS)
    df.to_csv(pathname + '_DMOS.csv', index=False)
    
    df_image_name = pd.DataFrame (image_name)
    df_image_name.to_csv(pathname + '_name_dmos.csv', index=False)