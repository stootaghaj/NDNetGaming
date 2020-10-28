# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:23:44 2020

@author: Saman Zadtootaghaj
"""


import ffmpeg
import  numpy as np
import imageio
np.random.seed(7)
import tensorflow as tf
tf.Session()
tf.set_random_seed(9)
from keras.models import  load_model
from keras.applications import densenet
import glob
import os
import pandas as pd
import argparse

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def tempo(TC, MOS):
    Pooled = -1.71 + 1.107 * MOS + 0.00053* TC**3- 0.024 * TC**2 + 0.353 * TC
    return Pooled.clip(min=1.1, max=4.91)

def test_video(model, videopath, videoname):    
  
    NDGmodel = load_model(model)        
    probe = ffmpeg.probe(videopath + videoname ) 
    video_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    out, err = (ffmpeg
                .input(videopath + videoname)
                .output('pipe:', format='rawvideo', pix_fmt = 'rgb24')
                .run(capture_stdout = True)
                )
    video = np.frombuffer(out, np.uint8).reshape([-1, height,width,3])
    
    
    MOS = np.empty(np.size(video,0))
    TC = np.empty(np.size(video,0))
    count = 0;
    for k in range(np.size(video,0)):
        ims = video[k,:,:,:]    
        patches = np.zeros((ims.shape[0]//299, ims.shape[1]//299, 299, 299, 3))
                
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patches[i,j] = ims[i*299:(i+1)*299, j*299:(j+1)*299]
                        
                        
        patches = densenet.preprocess_input(patches.reshape((-1, 299, 299, 3)))
        preds = NDGmodel.predict(patches)
        avgPred = np.mean(preds)
        MOS[count]= avgPred;
        if k==0:
            img_pre = ims
        TC[count] = (rgb2gray(ims)- rgb2gray(img_pre)).std()
        img_pre = ims;
        del avgPred, preds, patches;
        count = count + 1;
    print("MOS prediction based on temporal pooling:",np.mean(tempo(TC[:], MOS[:])[1:]))
    print("MOS prediction based on average pooling:",np.mean( MOS[:]));
    df = pd.DataFrame (MOS)
    df.to_csv(videoname + '_predicted_dmos.csv', index=False)



def test_image(model, folder, imageformat):   
    NDGmodel = load_model(model)
    
    Path =[x[0] for x in os.walk(folder)] 
    del Path[0]
    for pathname in Path:
        image_list = []
        image_name = []

        for filename in glob.glob(pathname + '/*.' + imageformat):
        
            im=np.array(imageio.imread(filename)) 
            image_list.append(im)
            image_name.append(filename)
        MOS = np.empty(len(image_list))
        count = 0;
        for ims in image_list:
            
            patches = np.zeros((ims.shape[0]//299, ims.shape[1]//299, 299, 299, 3))
            
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patches[i,j] = ims[i*299:(i+1)*299, j*299:(j+1)*299]
                    
                    
            patches = densenet.preprocess_input(patches.reshape((-1, 299, 299, 3)))
            preds = NDGmodel.predict(patches)
            avgPred = np.mean(preds)
            MOS[count]= avgPred;
            del avgPred, preds, patches;
            count = count + 1;
        print("MOS prediction based on average pooling:",np.mean( MOS[:]));
        name = pathname.split('/');
        df = pd.DataFrame (MOS)
        df.to_csv(name[-1] + '_predicted_dmos.csv', index=False)


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mp', '--model', action='store', dest='model', default=r'./models/subjectiveDemo2_DMOS_Final.model' ,
                    help='Specify model together with the path, e.g. ./models/subjectiveDemo2_DMOS_Final.model')
                    
    parser.add_argument('-vp', '--videopath', action='store', dest='videopath', default=r'./videos/' ,
                    help='Specify the path of video that is going to be evaluated')
                    
    parser.add_argument('-fr', '--videoname', action='store', dest='videoname', default='sample1.mp4' ,
                    help='Specify the name of the video e.g. sample.mp4')
    
    parser.add_argument('-fl', '--folder', action='store', dest='folder', default=r'./frames/' ,
                    help='the patch for the folder that contains folders of frames')
                    
    parser.add_argument('-imf', '--imageformat', action='store', dest='imageformat', default='png' ,
                    help='format of extracted frames, e.g. png, jpg, bmp')
    
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='video',
                    help='Option to select types of test, video or image_folders')
    
    values = parser.parse_args()

    if values.test_type == 'video':
        test_video(values.model, values.videopath, values.videoname);
        
    elif values.test_type == 'image_folders':
        test_image(values.model, values.folder, values.imageformat);
    
    else:
        print("No such option")
