# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:22:07 2019

@author: utke.markus
"""


from scipy import stats
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction import image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio
import cv2

from keras.callbacks import Callback
from keras.applications import densenet, xception, resnet50, mobilenet_v2, vgg19
from keras.layers import Dense
from keras.models import Model, load_model



class PatchPatterns:
    ''' For saving the slices that are used for the patch patterns
    
        patches:          slices grouped by position
        patchesPerNumber: slices grouped by number of patches
    '''
    patches = {
            'center':               [np.index_exp[540-149:540+150, 960-149:960+150]],
            'corners':              [np.index_exp[:299,:299],
                                     np.index_exp[-299:,:299],
                                     np.index_exp[:299,-299:],
                                     np.index_exp[-299:,-299:]],
            'nextToCenter':         [np.index_exp[540-149:540+150,555-149:555+150],
                                     np.index_exp[540-149:540+150,-555-149:-555+150]],
            'innerCorners':         [np.index_exp[195:195+299,555-149:555+150],
                                     np.index_exp[195:195+299,-555-149:-555+150],
                                     np.index_exp[-195-299:-195,555-149:555+150],
                                     np.index_exp[-195-299:-195,-555-149:-555+150]],
            'leftRight':            [np.index_exp[540-149:540+150,:299],
                                     np.index_exp[540-149:540+150,-299:]],
            'topBottom':            [np.index_exp[:299, 960-149:960+150],
                                     np.index_exp[-299:, 960-149:960+150]],
            'aboveUnderCenter':     [np.index_exp[540-149-299:540-149,960-149:960+150],
                                     np.index_exp[540+150:540+150+299,960-149:960+150]],
            'cornersNextToCenter':  [np.index_exp[540-299:540,960-149-299:960-149],
                                     np.index_exp[540-299:540,960+150:960+150+299],
                                     np.index_exp[540:540+299,960-149-299:960-149],
                                     np.index_exp[540:540+299,960+150:960+150+299]]
            }
    
    patchesPerNumber = {
            1:  patches['center'],
            5:  patches['center'] + patches['corners'],
            7:  patches['center'] + patches['corners'] + patches['nextToCenter'],
            9:  patches['center'] + patches['corners'] + patches['innerCorners'],
            11: patches['center'] + patches['corners'] + patches['innerCorners'] + patches['topBottom'],
            13: patches['center'] + patches['corners'] + patches['innerCorners'] + patches['topBottom'] + patches['leftRight'],
#            9:  patches['center'] + patches['corners'] + patches['cornersNextToCenter'],
#            11: patches['center'] + patches['corners'] + patches['cornersNextToCenter'] + patches['aboveUnderCenter'],
#            13: patches['center'] + patches['corners'] + patches['cornersNextToCenter'] + patches['aboveUnderCenter'] + patches['leftRight'],        
            }


def extractRandomPatches(img, nPatches, patchSize):
    '''     Returns a number of random patches from the given image
    
            Parameters:
                img:        image as numpy array \n
                nPatches:   number of patches to return \n
                patchSize:  size of the quadratic patches \n
    '''
    return image.extract_patches_2d(img, (patchSize,patchSize), max_patches=nPatches)

def extractPatchPattern(img, nPatches, patchSize):
    '''     Returns a number of patches from the given image following a specific pattern
            
            Parameters:
                img:        image of shape (1080,1920,3) as numpy array \n
                nPatches:   number of patches to return, either 1, 5, 7, 9, 11 or 13 \n
                patchSize:  size of the quadratic patches, for now only 299 is allowed \n
    '''
    if patchSize == 299:
        if nPatches in [1,5,7,9,11,13]:
            patches = np.zeros((nPatches, patchSize, patchSize, 3))
            
            for (i, patch) in enumerate(PatchPatterns.patchesPerNumber[nPatches]):
                patches[i] = img[patch]
            return patches
        else:
            raise Exception("Please choose 1, 5, 7, 9, 11 or 13 as number of patches")
    else:
        raise Exception("Please choose 299 as patch size")



def getMetrics(yPred, y, verbose = True):
    ''' Returns different metrics as a tuple (R^2, RMSE, PCC, SRCC) for comparing predictions with true values
            
        **Parameters**:
            * yPred:   Array with predictions
            * y:       Array with true values
            * verbose: Either 1 to print metrics to the console or 0 to only return them
    '''
    r2 = metrics.r2_score(y, yPred)
    rmse = np.sqrt(metrics.mean_squared_error(y, yPred))
    pcc = stats.pearsonr(y, yPred)[0]
    srcc = stats.spearmanr(y, yPred)[0]
    if verbose:
        print("R²: %1.4f \t RMSE: %1.4f \t PCC: %1.4f \t SRCC: %1.4f" % (r2, rmse, pcc, srcc))
    return r2, rmse, pcc, srcc



def plotPrediction(yPred, yVal, nValVideos, title = None):
    ''' Plots the predictions after training over the real values on frame and video level
            
        **Parameters**:
            * validationCallback:   Array with predictions
            * y:       Array with true values
            * verbose: Either 1 to print metrics to the console or 0 to only return them
    '''
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize = (7,8))
    plt.plot(yVal , yPred, 'co', markersize = .8) 
    plt.plot(np.mean(yVal.reshape((nValVideos,-1)), axis=-1),np.mean(yPred.reshape((nValVideos,-1)), axis=-1), 'go', markersize = 4)
    plt.plot(np.arange(101), 'k')
    plt.xlim((-10,110))
    plt.ylim((-10,110))
    plt.xlabel("Real VMAF")
    plt.ylabel("Predicted VMAF")
    plt.legend(["Frame level", "Video level"], loc = 2)
    plt.text(0, -35, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(yPred, yVal))
    plt.text(0, -42, "Video Level: R²: %1.2f   RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(np.mean(yPred.reshape((nValVideos,-1)), axis=-1), np.mean(yVal.reshape((nValVideos,-1)), axis=-1)))
    plt.subplots_adjust(bottom=.2)
    if(title):
        plt.savefig(title, bbox_inches = 'tight')
    
    
    
def plotPredictionSubjective(yPred, yVal, title = None):
    ''' Plots the predictions after training over the real values on frame and video level
            
        **Parameters**:
            * validationCallback:   Array with predictions
            * y:       Array with true values
            * verbose: Either 1 to print metrics to the console or 0 to only return them
    '''
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize = (7,8))
    plt.plot(yVal , yPred, 'co', markersize = 2.5) 
    plt.plot(np.arange(6), 'k')
    plt.xlim((-.5,5.5))
    plt.ylim((-.5,5.5))
    plt.xlabel("Real MOS")
    plt.ylabel("Predicted MOS")
    plt.text(0, -1.75, "Frame Level: R²: %1.2f  RMSE: %1.2f   PCC: %1.3f   SRCC: %1.3f" % getMetrics(yPred, yVal))
    plt.subplots_adjust(bottom=.2)
    if(title):
        plt.savefig(title, bbox_inches = 'tight')
    
    
def setTrainability(model, nTrainableLayers):
    ''' Sets the trainability of the last nTrainableLayers of model to True, sets it to False for all the layers before
            
        **Parameters**:
            * model:            Keras model
            * nTrainableLayers: Number of layers at the end of the model, that should still be trainable
    '''
    for layer in model.layers[:-nTrainableLayers]:
        layer.trainable = False
    
    for layer in model.layers[-nTrainableLayers:]:
        layer.trainable = True



class ValidationCallback(Callback):
    ''' Records the validation loss after each epoch
        can be passed as a callback to the keras fit method
        Saves the model with the best RMSE
        
        **Parameters**:
            * dsVal:      DataSequence for loading the validation data
            * nValVideos: Number of Videos in the validation set
            * title:      Title for the model that is being saved

    '''
    def __init__(self, dsVal, nValVideos, title):
        self.dsVal = dsVal
        self.nValVideos = nValVideos
        self.title = title
    
    def on_train_begin(self, logs={}):
        self.metricsPerFrame = []
        self.metricsPerVideo = []
        self.preds = []
        self.yVal = []
        for i in range(self.dsVal.__len__()):
            self.yVal.append(self.dsVal.__getitem__(i))
        self.yVal = np.concatenate(self.yVal)
        self.yValPerPic = self.yVal[::self.dsVal.nPatches]
        self.dsVal.switchValReturnMode()
        
    def on_epoch_end(self, epoch, logs={}):
        predVal = self.model.predict_generator(self.dsVal) 
        predValPerPic = np.mean(predVal.reshape((-1,self.dsVal.nPatches)), axis = -1)
        self.preds.append(predValPerPic)
        self.metricsPerFrame.append(getMetrics(predValPerPic, self.yValPerPic, verbose=True))
        self.metricsPerVideo.append(getMetrics(np.mean(predValPerPic.reshape((self.nValVideos,-1)), axis=-1),np.mean(self.yValPerPic.reshape((self.nValVideos,-1)), axis=-1), verbose=True))
        if np.argmin(np.array(self.metricsPerFrame)[:,1]) == epoch:
            print("New Best Epoch: %i" % (epoch+1))
            self.model.save(self.title +".model")
            self.bestEpoch = epoch


class ValidationCallbackSubjective(Callback):
    def __init__(self, xVal, yVal, nPatches, title):
        self.xVal = xVal
        self.yVal = yVal
        self.nPatches = nPatches
        self.title = title
    
    def on_train_begin(self, logs={}):
        self.metricsPerFrame = []
        self.preds = []
        self.yValPerPic = self.yVal[::self.nPatches]
        
    def on_epoch_end(self, epoch, logs={}):
        predVal = self.model.predict(self.xVal) 
        predValPerPic = np.mean(predVal.reshape((-1,self.nPatches)), axis = -1)
        self.preds.append(predValPerPic)
        self.metricsPerFrame.append(getMetrics(predValPerPic, self.yValPerPic, verbose=True))
        if np.argmin(np.array(self.metricsPerFrame)[:,1]) == epoch:
            print("New Best Epoch: %i" % (epoch+1))
            self.model.save(self.title +".model")
            self.bestEpoch = epoch



def buildModel(modelBuilder, patchSize):
    ''' Builds the model and returns it
            
        **Parameters**:
            * modelBuilder: Keras model constructor, e.g. keras.applications.densenet.DenseNet121
            * patchSize:    Size of the quadratic patches
    '''
    partModel = modelBuilder(include_top = False, weights = None, pooling = 'avg', input_shape = (patchSize,patchSize,3))
    weightModel = modelBuilder(include_top = False, weights = 'imagenet', pooling = 'avg', input_shape = (224,224,3))
    partModel.set_weights(weightModel.get_weights())
    outputs = partModel.output
    outputs = Dense(1, activation = 'linear')(outputs)
    return Model(inputs = partModel.input, outputs = outputs)




def plotLocalPredictions(model, modelBuilder, preprocessing, filepath, mos = None, nPatches = None, smooth = True, img_weights = (4,6,2), filename = "test"):
    ''' Plots the local predictions of a model for a given image
        Can also show the patch pattern on top of the image
            
        **Parameters**:
            * model:         Model with the trained weights
            * modelBuilder:  Keras model constructor that was used to build the model, e.g. keras.applications.densenet.DenseNet121
            * preprocessing: Preprocessing method to apply to the image, e.g. keras.applications.densenet.preprocess_input
            * filepath:      Path to the image
            * mos:           MOS value for the image. If None MOS is not shown
            * nPatches:      Number of patches to show and use. If None no patches are shown
            * smooth:        Whether to smooth the local predictions
    '''
    
    if nPatches and not nPatches in [1,5,7,9,11,13]:
        raise Exception("Please choose 1, 5, 7, 9, 11 or 13 as number of patches")

    # build model with full image input size and globalAverage and dense layer cut off
    largeModel = modelBuilder(include_top = False, weights = None, pooling = None, input_shape = (1080,1920,3))
    largeModel.set_weights(model.get_weights()[:-2])
    
    # load image
    img_plot = np.array(imageio.imread(filepath))
    img = preprocessing(img_plot)
    
    # use weights of dense layer to calculate local predictions (global average pooling is skipped)
    conv = largeModel.predict(np.expand_dims(img, axis=0))[0]
    
    weights = model.get_weights()[-2:]
    
    predLocal = np.sum(conv*weights[0][:,0], axis = -1)+weights[1]
    predFull = np.mean(predLocal)
    
    # predict patches
    if(nPatches):
        imagePatches = []
        for patch in PatchPatterns.patchesPerNumber[nPatches]:
            imagePatches.append(img[patch])
        predPatches = np.mean(model.predict(np.array(imagePatches)))
    
    # input: rgb image, output: grayscale image
    def rgb2gray(rgb):
        return np.repeat(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])[...,None],3,2)
        
    plt.rcParams.update({'font.size': 14})
    cm = plt.get_cmap('jet')
    pic_weight = img_weights[0]/(img_weights[0] + img_weights[1])
    
    # rescale local predictions to full image, make colored map from local predictions, add them
    pred_overlay = cv2.resize(predLocal, dsize=(1920,1080), interpolation= cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST)
    overlay = rgb2gray(img_plot/255.)*pic_weight
   
    colored_pred_overlay = cm(pred_overlay/6)[:,:,:3]
    
    overlay += colored_pred_overlay*(1-pic_weight)
    
    # plot 
    fig = plt.figure(figsize=(14,6.5))
    ax1 = fig.add_axes([0.00, 0.0, 0.82, 1.0])
    ax2 = fig.add_axes([0.895, 0.025, 0.025, 0.786])
    
    ax1.imshow(overlay)
    ax1.axis('off')
    
    # overlay patches
    if(nPatches):
        patchImage = np.zeros((1080,1920,3))
        patches = PatchPatterns.patchesPerNumber[nPatches]
        for patch in patches:
            patchImage[patch] = 1
        patchWeight =  img_weights[2]/(img_weights[0] + img_weights[1] + img_weights[2])
        ax1.imshow(np.multiply(overlay,patchImage*patchWeight+(1-patchWeight)))

    # show colorbar
    norm = mpl.colors.Normalize(vmin=-.5, vmax=5.5)
    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cm, norm=norm)
    
    cbar.set_ticks(np.array([0,1,2,3,4,5,6]))
    cbar.ax.minorticks_on()
    
    # plot on colorbar
    cbar.ax.plot([-.5,5.5], [predFull]*2, 'k')
    cbar.ax.plot([1.5], [predFull], 'kx', markersize = 7) 
     
    legend_elements = [Line2D([0], [0], marker='x', color='k', label='Image Prediction', markersize = 7)]

    if(nPatches in [1, 5, 7, 9, 11, 13]):   
        legend_elements.append(Line2D([0], [0], marker='o', color='k', label='Patch Prediction', markersize = 7))
        cbar.ax.plot([-.5,5.5], [predPatches]*2, 'k')
        cbar.ax.plot([4.0], [predPatches], 'ko', markersize = 7) 
    if(mos):
        cbar.ax.plot([-.5,5.5], [mos]*2, 'w')
        legend_elements.append(Line2D([0], [0], color='w', label='Fragmentation value'))

    # add legend
    leg = cbar.ax.legend(handles=legend_elements, bbox_to_anchor=(4.42, 1.23))
    leg.get_frame().set_facecolor('#BBBBBB')
    fig.tight_layout()
    plt.savefig(filename + ".png")
    
            
    
def countLayers(model, nLastConvolutionalLayers):
    layers = model.layers
    convCounter = 0
    for i in range(1, len(layers)+1):
        if convCounter == nLastConvolutionalLayers:
            return i-1
        if type(layers[-i]).__name__ == "Conv2D" or type(layers[-i]).__name__ == "SeparableConv2D":
            convCounter += 1
    return len(layers)
            

#
#preprocessList = [xception.preprocess_input, resnet50.preprocess_input, densenet.preprocess_input, mobilenet_v2.preprocess_input]
#modelBuilderList = [xception.Xception, resnet50.ResNet50, densenet.DenseNet121, mobilenet_v2.MobileNetV2]
#
#modelNumber = 2
#
#partModel = modelBuilderList[modelNumber](include_top = False, weights = None, pooling = 'avg', input_shape = (299,299,3))
#weightModel = modelBuilderList[modelNumber](include_top = False, weights = None, pooling = 'avg', input_shape = (224,224,3))
#partModel.set_weights(weightModel.get_weights())
#outputs = partModel.output
#outputs = Dense(1, activation = 'linear')(outputs)
#model = Model(inputs = partModel.input, outputs = outputs)
#model.summary()
#
#layers = model.layers
#
#
#
##%%
#
##model.summary()
#
#c = 0
#for i in range(len(model.layers)):
#    if model.layers[-i].name == "k":
#        break;
#    else:
#        c += 1