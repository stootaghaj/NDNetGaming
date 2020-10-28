#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:54:29 2020

@author: saman
"""

#from __future__ import division

import numpy as np
from ffprobe import FFProbe
import argparse
import re
import pandas as pd

def test_model(bitrate, coding_res, framerate, clss):

# set the coefficients    
    VDcoef=[[66.29, 10.23],
      [67.28, 3.23],  
    [67.28, 3.23]];
    
    VFcoef = [[2.1, -5.426, 0.0005258],
             [13.79, -8.017, 0.0005442],
     [11.21, -10.59, 0.0006314]];
    
    VUcoef = [[4.299, -2.016, -17.99], 
              [18.58, -3.422, -15.38],
              [17.13, -4.494, -7.844]];          
              
  
#transform data from MOS to R  for one value
    def MOSfromR_Value(Q):
         MOS_MAX = 4.64;#np.max(df['MOS']);#4.5;#np.max(df['MOS']); #4.9
         MOS_MIN = 1.3;#np.min(df['MOS']); #; #np.min(df['MOS']); #1.05
         MOS = MOS_MIN + (MOS_MAX-MOS_MIN)/100*Q + Q*(Q-60)*(100-Q)* 7.0e-6
         return MOS
       
#transform data from MOS to R  for an array of values

    def MOSfromR(Q):
         MOS = np.zeros(Q.shape)
         MOS_MAX = 4.64;
         MOS_MIN = 1.3;
         for i in range(len(Q)):
              if (Q[i] > 0 and Q[i] < 100):
                   MOS[i] = MOS_MIN + (MOS_MAX-MOS_MIN)/100*Q[i] + Q[i]*(Q[i]-60)*(100-Q[i])* 7.0e-6
              elif (Q[i] >= 100):
                   MOS[i] = MOS_MAX
              else:
                   MOS[i] = MOS_MIN
         return MOS
    
    
    def VF(bitrate, framerate, coding_res, coeff):
       
        bitperpixel = bitrate/(framerate*coding_res)
        IVF = coeff[0] + coeff[1]*np.log(bitperpixel*bitrate) + coeff[2]*bitrate
        return IVF.clip(min=0, max=68.52)
    
    
    def VU(bitrate, framerate, coding_res, coeff):
        
        bitperpixel = bitrate/(framerate*coding_res)
        scaleratio = coding_res/(1080*1920)
        IVU = coeff[0] + coeff[1]*np.log(bitperpixel*bitrate) + coeff[2]*np.log(scaleratio)
        return IVU.clip(min=0, max=67.90)
    
    def VD(framerate, coeff):
        
        IVD = np.exp(coeff[0]/framerate) + coeff[1]
        return IVD.clip(min=0,max=70)
        
    #choose the righ coef for the 1071 based on the class complexity--- and calling IVD and IQ --- the coef has nothing to do with IQ and IVD    
        
    if clss=='Low':
 
        I_VD = VD(framerate,VDcoef[0]);
        I_VU = VU(bitrate, framerate, coding_res,VUcoef[0]);
        I_VF = VF(bitrate, framerate, coding_res,VFcoef[0]);
        
    elif clss=='Medium':
    
        I_VD = VD(framerate, VDcoef[1]);
        I_VU = VU(bitrate, framerate, coding_res,VUcoef[1]);
        I_VF = VF(bitrate, framerate, coding_res,VFcoef[1]);
    
    elif clss=='High':
            
        I_VD = VD(framerate,VDcoef[2]);
        I_VU = VU(bitrate, framerate, coding_res,VUcoef[2]);
        I_VF = VF(bitrate, framerate, coding_res,VFcoef[2]);
          
    R_QoE = 100 - 0.259*I_VD - 0.554*I_VF - 0.341*I_VU; 
    print("Video Quality:",MOSfromR_Value(R_QoE))  
    print("Video discontinuity:", MOSfromR_Value(100-I_VD)) ;
    print("Video unclearness:", MOSfromR_Value(100-I_VU)) ;
    print("Video fragmentation:", MOSfromR_Value(100-I_VF)) ;



def test_video(video, complexity):
    
    metadata=FFProbe(video)
    coding_res = int(metadata.video[0].height)*int(metadata.video[0].width)
    bitrate = int(re.search(r'\d+', metadata.metadata['bitrate']).group());
    framerate = int(metadata.video[0].framerate);
    test_model(bitrate, coding_res, framerate, complexity)
    
 

def test_para(bitrate, coding_res, framerate, complexity):
    wh = coding_res.split('x');
    dim = int(wh[0])*int(wh[1]);
    test_model(bitrate, dim, framerate, complexity);


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-br', '--bitrate', action='store', dest='bitrate', default=1000 ,
                    help='Specify the bitrate of video', type=int)
                    
    parser.add_argument('-re', '--coding_res', action='store', dest='coding_res', default='1920x1080' ,
                    help='Specify the coding resulotion of video')
                    
    parser.add_argument('-fr', '--framerate', action='store', dest='framerate', default=30 ,
                    help='Specify the framerate of video', type=int)
    
    parser.add_argument('-clss', '--complexity', action='store', dest='complexity', default='High' ,
                    help='Specify the class of model, Low, Medium, High')
                    
    parser.add_argument('-vid', '--video', action='store', dest='video', default=r'CSGO_30fps_30sec_Part1_640x480_400_x264.mp4' ,
                    help='Number of Images')
    
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='video',
                    help='Option to')
    
    values = parser.parse_args()

    if values.test_type == 'video':
        test_video(values.video, values.complexity);
        
    elif values.test_type == 'parameters':
        test_para(values.bitrate, values.coding_res, values.framerate, values.complexity);
        
    else:
        print("No such option")