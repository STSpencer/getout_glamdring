from __future__ import absolute_import, division, print_function

import matplotlib as mpl
import numpy as np
import keras
import os, tempfile, sys, glob, h5py
from keras.utils import HDF5Matrix
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, GaussianNoise, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation, Dropout
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from  matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
#from ctapipe.image import tailcuts_clean
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from skimage.restoration import denoise_wavelet, denoise_bilateral
import glob
global trainevents
trainevents = []
global validevents
validevents = []
global testevents
testevents = []
global train2
train2 = []
global test2
test2 = []
global valid2
valid2 = []

def get_confusion_matrix_one_hot(runname,model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    mr=[]
    mr2=[]
    mr3=[]
    nomod=np.shape(model_results)[0]
    notruth=np.shape(truth)[0]
    noev=min([nomod,notruth])
    for x in model_results:
        mr.append(np.argmax(x))
        mr2.append(x)
        if np.argmax(x)==0:
            mr3.append(1-x[np.argmax(x)])
        elif np.argmax(x)==1:
            mr3.append(x[np.argmax(x)])
    model_results=np.asarray(mr)[:noev]
    truth=np.asarray(truth)[:noev]
    mr2=mr2[:noev]
    mr3=mr3[:noev]

    cm=confusion_matrix(y_target=truth,y_predicted=np.rint(np.squeeze(model_results)),binary=True)
    fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(5,5))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/home/spencers/Figures/'+runname+'confmat.png')
    fpr,tpr,thresholds=roc_curve(truth,np.asarray(mr2)[:,1])
    plt.figure()
    lw = 2
    aucval=auc(fpr,tpr)
    fpr=np.asarray(fpr)
    tpr=np.asarray(tpr)
    print(aucval)
    fpr=1.0-fpr
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % aucval)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    #plt.legend(loc="lower right")
    plt.savefig('/home/spencers/Figures/'+runname+'_sigeff.png')
    np.save('/home/spencers/confmatdata/'+runname+'_sigef.npy',fpr)
    np.save('/home/spencers/confmatdata/'+runname+'_bgrej.npy',tpr)
    return cm

def hardcode_valid():
    batchflag='Valid'
    hexmethod='oversampling'
    onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))
    batch_size=20
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total
    global trainevents
    global testevents
    global train2
    global test2
    if batchflag == 'Train':
        filelist = onlyfiles[:10]
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[20:30]
        global testevents
        global test2
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            testevents = testevents + inputdata['isGamma'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Valid':
        filelist = onlyfiles[10:20]
        global validevents
        global valid2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            validevents = validevents + inputdata['isGamma'][:].tolist()
            valid2 = valid2 + inputdata['id'][:].tolist()
            inputdata.close()
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt
    
    while True:
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainarr = np.asarray(inputdata[hexmethod][:, :, :, :])
            labelsarr = np.asarray(inputdata['isGamma'][:])
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            notrigs=np.shape(trainarr)[0]
            
            for x in np.arange(np.shape(trainarr)[0]):
                chargevals = []
                for y in np.arange(4):
                    chargevals.append(np.sum(trainarr[x,y,:,:]))

                chargevals = np.argsort(chargevals)
                chargevals = np.flip(chargevals,axis=0) #Flip to descending order.
                trainarr[x, :, :, :] = trainarr[x, chargevals, :, :]
                #Denoising
                for y in np.arange(4):
                    im1=trainarr[x,y,:,:]
                    im1=im1[:,:,0]
                    im1=im1+abs(np.min(im1))
                    im2=denoise_bilateral(im1,multichannel=False)
                    im2=np.expand_dims(im2,axis=2)
                    trainarr[x,y,:,:]=im2
                    
            training_sample_count = len(trainarr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + 1000
            countarr = np.arange(0, len(labelsarr))

#            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                X = np.nan_to_num(X)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=2)
                yield (np.array(X), np.array(Y))

def hardcode_train():
    batchflag='Train'
    hexmethod='oversampling'
    onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))
    batch_size=20
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total
    global trainevents
    global testevents
    global train2
    global test2
    if batchflag == 'Train':
        filelist = onlyfiles[:10]
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[20:30]
        global testevents
        global test2
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            testevents = testevents + inputdata['isGamma'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Valid':
        filelist = onlyfiles[20:30]
        global validevents
        global valid2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            validevents = validevents + inputdata['isGamma'][:].tolist()
            valid2 = valid2 + inputdata['id'][:].tolist()
            inputdata.close()
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt
    
    while True:
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainarr = np.asarray(inputdata[hexmethod][:, :, :, :])
            labelsarr = np.asarray(inputdata['isGamma'][:])
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            notrigs=np.shape(trainarr)[0]
            
            for x in np.arange(np.shape(trainarr)[0]):
                chargevals = []
                for y in np.arange(4):
                    chargevals.append(np.sum(trainarr[x,y,:,:]))

                chargevals = np.argsort(chargevals)
                chargevals = np.flip(chargevals,axis=0) #Flip to descending order.
                trainarr[x, :, :, :] = trainarr[x, chargevals, :, :]
                #Denoising
                for y in np.arange(4):
                    im1=trainarr[x,y,:,:]
                    im1=im1[:,:,0]
                    im1=im1+abs(np.min(im1))
                    im2=denoise_bilateral(im1,multichannel=False)
                    im2=np.expand_dims(im2,axis=2)
                    trainarr[x,y,:,:]=im2
                    
            training_sample_count = len(trainarr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + 1000
            countarr = np.arange(0, len(labelsarr))

#            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                X = np.nan_to_num(X)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=2)
                yield (np.array(X), np.array(Y))

def generate_training_sequences(onlyfiles,batch_size, batchflag,hexmethod):
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total
    global trainevents
    global testevents
    global train2
    global test2
    if batchflag == 'Train':
        filelist = onlyfiles[:120]
        print('train', filelist)
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[120:160]
        print('test', filelist)
        global testevents
        global test2
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            testevents = testevents + inputdata['isGamma'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Valid':
        filelist = onlyfiles[160:]
        print('valid', filelist)
        global validevents
        global valid2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            validevents = validevents + inputdata['isGamma'][:].tolist()
            valid2 = valid2 + inputdata['id'][:].tolist()
            inputdata.close()
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt
    
    while True:
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainarr = np.asarray(inputdata[hexmethod][:, :, :, :])
            labelsarr = np.asarray(inputdata['isGamma'][:])
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            notrigs=np.shape(trainarr)[0]
            
            for x in np.arange(np.shape(trainarr)[0]):
                chargevals = []
                for y in np.arange(4):
                    chargevals.append(np.sum(trainarr[x,y,:,:]))

                chargevals = np.argsort(chargevals)
                chargevals = np.flip(chargevals,axis=0) #Flip to descending order.
                trainarr[x, :, :, :] = trainarr[x, chargevals, :, :]
                #Denoising
                for y in np.arange(4):
                    im1=trainarr[x,y,:,:]
                    im1=im1[:,:,0]
                    im1=im1+abs(np.min(im1))
                    im2=denoise_bilateral(im1,multichannel=False)
                    im2=np.expand_dims(im2,axis=2)
                    trainarr[x,y,:,:]=im2
                    
            training_sample_count = len(trainarr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + 1000
            countarr = np.arange(0, len(labelsarr))

#            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                X = np.nan_to_num(X)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=2)
                yield (np.array(X), np.array(Y))

def generate_real_sequences(onlyfiles,batch_size,hexmethod):
    """ Generates real test sequences on demand
    """
    nofiles = 0
    i = 0  # No. events loaded in total
    global testevents
    global test2
    filelist = onlyfiles
    print('test', filelist)
    for file in filelist:
        try:
            inputdata = h5py.File(file, 'r')
        except OSError:
            continue
        test2 = test2 + inputdata['id'][:].tolist()
        inputdata.close()

    while True:
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainarr = np.asarray(inputdata[hexmethod][:, :, :, :])
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            notrigs=np.shape(trainarr)[0]
            
            for x in np.arange(np.shape(trainarr)[0]):
                chargevals = []
                for y in np.arange(4):
                    chargevals.append(np.sum(trainarr[x,y,:,:]))

                chargevals = np.argsort(chargevals)
                chargevals = np.flip(chargevals,axis=0) #Flip to descending order.
                
                trainarr[x, :, :, :] = trainarr[x, chargevals, :, :]

                            
            training_sample_count = len(trainarr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + len(idarr)
            countarr = np.arange(0, len(idarr))

#            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                events=idarr[batch_idxs]
                Y=events.tolist()
                X = np.nan_to_num(X)
                yield (np.array(X),Y)
