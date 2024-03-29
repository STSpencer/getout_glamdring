'''Uses Keras to test a 2dconvlstm on VERITAS data.
Written by S.T. Spencer 8/7/2019'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
from keras.models import load_model
import numpy as np
import h5py
import keras
import os
import tempfile
import sys
from keras.utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization, Conv3D, GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input, GaussianNoise
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K
import tensorflow as tf
from keras.utils import plot_model
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from matplotlib.pyplot import cm
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
from sklearn.metrics import roc_curve, auc

import sys
plt.ioff()

# Finds all the hdf5 files in a given directory
global realdata
print('Started')
eventnumbers=[]
realdata = sorted(glob.glob('/mnt/extraspace/exet4487/Real64080/*.hdf5'))
runname = str(sys.argv[1])
runcode = 64080
hexmethod='oversampling'
#modfile='/users/exet4487/hypermodels/'+str(runname)+'.h5' #Model file to use For hypermodel selection
modfile='/mnt/extraspace/exet4487/ai40p1ln3885525519_checkpoint/'+str(runname) #For running checkpoints with longtrain
evlist=[]

for file in realdata:
    try:
        h5file=h5py.File(file, 'r')
    except OSError:
        realdata.remove(file)
        continue
    events=h5file['id'][:].tolist()
    evlist=evlist+events
    h5file.close()
    
def generate_real_sequences(onlyfiles,batch_size,hexmethod):
    """ Generates real test sequences on demand"""
    nofiles = 0
    i = 0  # No. events loaded in total
    global testevents
    global test2
    test2=[]
    filelist = onlyfiles
    print('test', filelist)
    for file in filelist:
        inputdata = h5py.File(file, 'r')
        test2 = test2 + inputdata['id'][:].tolist()
        inputdata.close()
    while True:
        for file in filelist:
            inputdata = h5py.File(file, 'r')
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
            if remainder_samples:
                batches = batches + 1
            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx * batch_size:idx * batch_size + batch_size]
                X = trainarr[batch_idxs]
                events=idarr[batch_idxs]
                #Y = events.tolist()
                X = np.nan_to_num(X)
                yield np.array(X)


def generate_real_evnos(onlyfiles,batch_size,hexmethod):
    """ Generates real test sequences on demand"""
    nofiles = 0
    i = 0  # No. events loaded in total                                                                                                                                                                                                                                                                                                                        
    global testevents
    filelist = onlyfiles
    print('test', filelist)
    while True:
        for file in filelist:
            inputdata = h5py.File(file, 'r')
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
            if remainder_samples:
                batches = batches + 1
            # generate batches of samples                                                                                                                                                                                                                                                                                                                      
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx * batch_size:idx * batch_size + batch_size]

                events=idarr[batch_idxs]
                Y = events.tolist()                                                                                                                                                                                                                                                                                                                          
                yield np.array(Y)
                
#noev=100
noev=len(evlist)
batchsize=1
#no_steps=int(noev/float(batchsize))
no_steps=noev
print('No Steps:',no_steps)

global Trutharr

print('No_events:',noev,'No_steps',no_steps)

# Define model architecture.
if hexmethod in ['axial_addressing','image_shifting']:
    inpshape=(None,27,27,1)
elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversampling','rebinning']:
    inpshape=(None,54,54,1)
else:
    print('Invalid Hexmethod')
    raise KeyboardInterrupt

# Test the network
print('Predicting')
model=load_model(modfile)
g2=generate_real_sequences(realdata,batchsize,hexmethod)
pred = model.predict(g2,verbose=1,workers=0,use_multiprocessing=False,steps=noev)
p2=[]
for i in np.arange(np.shape(pred)[0]):
    score=np.argmax(pred[i])
    if score==0:
        s2=1-pred[i][0]
    elif score==1:
        s2=pred[i][1]
    p2.append(s2)
p2=np.asarray(p2)
np.save('/users/exet4487/predictions/'+str(runcode)+'_'+runname+'_predictions_REAL.npy', p2)

gen=generate_real_evnos(realdata,batchsize,hexmethod)

for i in np.arange(no_steps):
    outp=next(gen)
    eventnumbers.append(outp[0])

eventnumbers=np.asarray(eventnumbers)
np.save('/users/exet4487/events/'+str(runcode)+'_'+runname+'_eventnos_REAL.npy', eventnumbers)
print(len(pred),eventnumbers,np.shape(eventnumbers),len(eventnumbers))
