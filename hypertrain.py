'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''
import os
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import h5py
import keras
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

import hyperas
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe, mongoexp
import pickle
import tempfile

plt.ioff()

# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
runname = 'hyperasglamdringtest2'
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

def data():
    def hardcode_valid():
        hexmethod='oversampling'
        onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
        batch_size=50
        """ Generates training/test sequences on demand
        """
        
        nofiles = 0
        i = 0  # No. events loaded in total
        filelist = onlyfiles[10:20]
        global validevents
        global valid2
        validevents=[]
        valid2=[]
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            validevents = validevents + inputdata['isGamma'][:].tolist()
            valid2 = valid2 + inputdata['id'][:].tolist()
            inputdata.close()
    
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
                training_sample_count = len(trainarr)
                batches = int(training_sample_count / batch_size)
                remainder_samples = training_sample_count % batch_size
                i = i + 1000
                countarr = np.arange(0, len(labelsarr))
                #trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
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
        hexmethod='oversampling'
        onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
        batch_size=50
        """ Generates training/test sequences on demand
        """
        
        nofiles = 0
        i = 0  # No. events loaded in total
        global trainevents
        global train2
        trainevents=[]
        train2=[]
        filelist = onlyfiles[:10]
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()
    
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
    history = model.fit(
        train_generator,
        steps_per_epoch=lentrain/50.0,
        epochs=1,
        verbose=0,
        workers=0,
        use_multiprocessing=False,
        shuffle=True,validation_data=validation_generator,validation_steps=lentruth/50.0)
    score, acc=model.evaluate(validation_generator,steps=lentruth/50.0)
    out = {'loss': -acc,
        'score': score,
        'status': STATUS_OK,
        'model_params': model.summary()
    }
    # optionally store a dump of your model here so you can get it from the database later                                                                                                   
    temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
    model.save(temp_name)
    with open(temp_name, 'rb') as infile:
        model_bytes = infile.read()
    out['model_serial'] = model_bytes
    return out

    # Plot training accuracy/loss.

trialsinit=mongoexp.MongoTrials('mongo://exet4487:admin123@192.168.0.200:27017/jobs/jobs',exp_key=runname)

run,model=optim.minimize(model=create_model,data=data,algo=tpe.suggest,max_evals=300,trials=trialsinit,keep_temp=True)

print('best run:', run)
print(trialsinit)
print(dir(trialsinit))
print(len(trialsinit))
print("----------trials-------------")
for i in trialsinit.trials:
    vals = i.get('misc').get('vals')
    results = i.get('result').get('loss')
    print(vals,results)
#pickle.dump(trialsinit, open(trialsfile, "wb"))
