'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import h5py
import keras
import tempfile
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

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
import tensorflow.python.keras.backend as K
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
from os import path
plt.ioff()

# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
runname = str(sys.argv[1])
hexmethod='oversampling'
homedir='/users/exet4487/'
trialsfile=homedir+'trials/'+runname+'.npy'

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
        filelist = onlyfiles[-20:]
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
        filelist = onlyfiles[:20]
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
    strategy=tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = Sequential()
        model.add(ConvLSTM2D(filters={{choice([10,20,30,40,50,60])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)])}},
                             input_shape=inpshape,
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2({{uniform(0,1)}}),dropout={{uniform(0,1)}},recurrent_dropout={{uniform(0,1)}}))
        model.add(BatchNormalization())
        
        model.add(ConvLSTM2D(filters={{choice([10,20,30,40,50,60])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)])}},
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2({{uniform(0,1)}}),dropout={{uniform(0,1)}},recurrent_dropout={{uniform(0,1)}}))
        model.add(BatchNormalization())
        
        model.add(ConvLSTM2D(filters={{choice([10,20,30,40,50,60])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)])}},
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2({{uniform(0,1)}}),dropout={{uniform(0,1)}},recurrent_dropout={{uniform(0,1)}}))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters={{choice([10,20,30,40,50,60])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)])}},
                             padding='same', return_sequences=True,dropout={{uniform(0,1)}}))
        model.add(BatchNormalization())
        
        if {{choice(['l5','no'])}}=='l5':
            model.add(ConvLSTM2D(filters={{choice([10,20,30,40,50,60])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)])}},
                                 padding='same', return_sequences=True,dropout={{uniform(0,1)}}))
            model.add(BatchNormalization())
            
        if {{choice(['l6','no'])}}=='l6':
            model.add(ConvLSTM2D(filters={{choice([10,20,30,40,50,60])}}, kernel_size={{choice([(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)])}},
                                 padding='same', return_sequences=True,dropout={{uniform(0,1)}}))
            model.add(BatchNormalization())

        model.add(GlobalAveragePooling3D())
        
        if {{choice(['l7','no'])}}=='l7':
            model.add(Dense({{choice([10,50,100,200])}},activation='relu'))
            model.add(Dropout({{uniform(0,1)}}))
        if {{choice(['l8','no'])}}=='l8':
            model.add(Dense({{choice([10,50,100,200])}},activation='relu'))
            model.add(Dropout({{uniform(0,1)}}))
        if {{choice(['l9','no'])}}=='l9':
            model.add(Dense({{choice([10,50,100,200])}},activation='relu'))
            model.add(Dropout({{uniform(0,1)}}))
        model.add(Dense(2, activation='softmax'))
        # Compile the model
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['binary_accuracy'])
        '''early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='auto')'''
    
    # Code for ensuring no contamination between training and test data.
    lentrain=19574*2
    lentruth=19600*2
# Train the network
    history = model.fit(
        train_generator,
        steps_per_epoch=lentrain/50.0,
        epochs=5,
        verbose=0,
        workers=0,
        use_multiprocessing=False,
        shuffle=True,validation_data=validation_generator,validation_steps=lentruth/50.0)
    print(history.history)
    print(history.history.keys())
    acc=np.amax(history.history['val_binary_accuracy']) 
    modelnumber=next(tempfile._get_candidate_names())
    modelcode=np.random.randint(0,1e10)
    out = {'loss': -acc,
        'modelno':str(modelnumber)+str(modelcode),
        'status': STATUS_OK,
        'model_params': model.summary()
    }
    model.save('/users/exet4487/hypermodels/'+str(modelnumber)+str(modelcode)+'.h5')
    # optionally store a dump of your model here so you can get it from the database later                                                                                                   
    temp_name = tempfile.gettempdir()+'/'+modelnumber + '.h5'
    model.save(temp_name)
    #with open(temp_name, 'rb') as infile:
    #model_bytes = infile.read()
    #out['model_serial'] = model_bytes
    return out

    # Plot training accuracy/loss.

trialsinit=mongoexp.MongoTrials('mongo://exet4487:admin123@192.168.0.200:27017/jobs/jobs',exp_key=runname)

run,model=optim.minimize(model=create_model,data=data,algo=tpe.suggest,max_evals=1010,trials=trialsinit,keep_temp=True)

print('best run:', run)
print(trialsinit)
print(dir(trialsinit))
print(len(trialsinit))
print("----------trials-------------")
trialsdict={}
for i in trialsinit.trials:
    vals = i.get('misc').get('vals')
    results = i.get('result').get('loss')
    print(vals,results)
    trialsdict[str(vals)]=str(results)
np.save(trialsfile,trialsdict)
#pickle.dump(trialsinit, open(trialsfile, "wb"))
