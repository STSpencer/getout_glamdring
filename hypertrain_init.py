'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''

import matplotlib as mpl
mpl.use('Agg')
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
from keras.backend.tensorflow_backend import set_session
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
from net_utils import *
import hyperas
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
import pickle

plt.ioff()

# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
runname = 'hyperasglamdringtest1'
hexmethod='oversampling'
homedir='/users/exet4487/'
trialsfile=homedir+'trials/'+runname+'.p'

global Trutharr
Trutharr = []
Train2=[]
truid=[]
print(onlyfiles,len(onlyfiles))

# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[20:30]:
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        continue
    labelsarr = np.asarray(inputdata['isGamma'][:])
    idarr = np.asarray(inputdata['id'][:])
    for value in labelsarr:
        Trutharr.append(value)
    for value in idarr:
        truid.append(value)
    inputdata.close()

for file in onlyfiles[:10]:
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        continue
    labelsarr = np.asarray(inputdata['isGamma'][:])
    for value in labelsarr:
        Train2.append(value)
    inputdata.close()

print('lentruth', len(Trutharr))
print('lentrain',len(Train2))
global lentrain
global lentruth
lentrain=len(Train2)
lentruth=len(Trutharr)
np.save(homedir+'truesim/truthvals_'+runname+'.npy',np.asarray(Trutharr))
np.save(homedir+'idsim/idvals_'+runname+'.npy',np.asarray(truid))

# Define model architecture.
if hexmethod in ['axial_addressing','image_shifting']:
    inpshape=(None,27,27,1)
elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversampling','rebinning']:
    inpshape=(None,54,54,1)
else:
    print('Invalid Hexmethod')
    raise KeyboardInterrupt

def data(onlyfiles,hexmethod):
    train_generator=hardcode_train()
    validation_generator=hardcode_valid()
    return train_generator, validation_generator

def create_model(train_generator,validation_generator):
    inpshape=(None,54,54,1)

    model = Sequential()
    model.add(ConvLSTM2D(filters={{choice([10,20,30,40])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5)])}},
                         input_shape=inpshape,
                         padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2({{uniform(0,1)}}),dropout={{uniform(0,1)}},recurrent_dropout={{uniform(0,1)}}))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters={{choice([10,20,30,40])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5)])}},
                         padding='same', return_sequences=True,dropout={{uniform(0,1)}},recurrent_dropout={{uniform(0,1)}},kernel_regularizer=keras.regularizers.l2({{uniform(0,1)}})))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters={{choice([10,20,30,40])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5)])}},
                         padding='same', return_sequences=True,dropout={{uniform(0,1)}}))
    model.add(BatchNormalization())
    if {{choice(['three','four'])}}=='four':
        model.add(ConvLSTM2D(filters={{choice([10,20,30,40])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5)])}},
    padding='same', return_sequences=True,dropout={{uniform(0,1)}}))
        model.add(BatchNormalization())

    model.add(BatchNormalization())
    model.add(GlobalAveragePooling3D())
    model.add(Dense({{choice([10,50,100,200])}},activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=['binary_accuracy'])
    
    '''early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')'''
    
    # Code for ensuring no contamination between training and test data.
    lentrain=19574
    lentruth=19600
# Train the network
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=lentrain/20.0,
        epochs=1,
        verbose=2,
        workers=0,
        use_multiprocessing=False,
        shuffle=True,validation_data=validation_generator,validation_steps=lentruth/20.0)
    score, acc=model.evaluate_generator(validation_generator,steps=lentruth/20.0)
    return{'loss':-acc,'status': STATUS_OK, 'model':model}

    # Plot training accuracy/loss.

train_generator,validation_generator=data(onlyfiles,hexmethod)
trialsinit=Trials()

run,model=optim.minimize(model=create_model,data=data,algo=tpe.suggest,max_evals=200,trials=trialsinit)
print('best run:', run)
print(len(trials))
pickle.dump(trialsinit, open(trialsfile, "wb"))
