#coding=utf-8

try:
    import os
except:
    pass

try:
    import matplotlib as mpl
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import h5py
except:
    pass

try:
    import keras
except:
    pass

try:
    import tempfile
except:
    pass

try:
    import sys
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    from tensorflow.compat.v1 import ConfigProto
except:
    pass

try:
    from tensorflow.compat.v1 import Session
except:
    pass

try:
    from keras.utils import HDF5Matrix
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization, Conv3D, GlobalAveragePooling3D
except:
    pass

try:
    from keras.layers.normalization import BatchNormalization
except:
    pass

try:
    from keras.layers.convolutional import AveragePooling2D
except:
    pass

try:
    from keras.layers.core import Activation
except:
    pass

try:
    from keras.layers.core import Dropout
except:
    pass

try:
    from keras.layers import Input, GaussianNoise
except:
    pass

try:
    from keras.models import Model
except:
    pass

try:
    from keras.layers import concatenate
except:
    pass

try:
    import tensorflow.python.keras.backend as K
except:
    pass

try:
    from keras.utils import plot_model
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    import glob
except:
    pass

try:
    from sklearn.preprocessing import StandardScaler
except:
    pass

try:
    from keras.callbacks import EarlyStopping
except:
    pass

try:
    from keras import regularizers
except:
    pass

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except:
    pass

try:
    import numpy.ma as ma
except:
    pass

try:
    from matplotlib.pyplot import cm
except:
    pass

try:
    from sklearn.preprocessing import scale
except:
    pass

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except:
    pass

try:
    import numpy.ma as ma
except:
    pass

try:
    from matplotlib.pyplot import cm
except:
    pass

try:
    from mlxtend.evaluate import confusion_matrix
except:
    pass

try:
    from mlxtend.plotting import plot_confusion_matrix
except:
    pass

try:
    from keras.metrics import binary_accuracy
except:
    pass

try:
    from sklearn.metrics import roc_curve, auc
except:
    pass

try:
    import hyperas
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe, mongoexp
except:
    pass

try:
    import pickle
except:
    pass

try:
    import tempfile
except:
    pass

try:
    from os import path
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def hardcode_valid():
    hexmethod='oversampling'
    onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080point/*.hdf5'))
    batch_size=50
    """ Generates training/test sequences on demand
    """
    
    nofiles = 0
    i = 0  # No. events loaded in total
    filelist = onlyfiles[-10:]
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
    onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080point/*.hdf5'))
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


def keras_fmin_fnct(space):

    inpshape=(None,54,54,1)
    strategy=tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = Sequential()
        model.add(ConvLSTM2D(filters=space['filters'], kernel_size=space['kernel_size'],
                             input_shape=inpshape,
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2']),dropout=space['l2_1'],recurrent_dropout=space['l2_2']))
        model.add(BatchNormalization())
        
        model.add(ConvLSTM2D(filters=space['filters_1'], kernel_size=space['kernel_size_1'],
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2_3']),dropout=space['l2_4'],recurrent_dropout=space['l2_5']))
        model.add(BatchNormalization())
        
        model.add(ConvLSTM2D(filters=space['filters_2'], kernel_size=space['kernel_size_2'],
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2_6']),dropout=space['l2_7'],recurrent_dropout=space['l2_8']))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=space['filters_3'], kernel_size=space['kernel_size_3'],
                             padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2_9']),dropout=space['l2_10'],recurrent_dropout=space['l2_11']))
        model.add(BatchNormalization())
        if space['l2_12']=='five':
            model.add(ConvLSTM2D(filters=space['filters_4'], kernel_size=space['kernel_size_4'],
                                 padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2_13']),dropout=space['l2_14'],recurrent_dropout=space['l2_15']))
            model.add(BatchNormalization())
        if space['l2_16']=='six':
            model.add(ConvLSTM2D(filters=space['filters_5'], kernel_size=space['kernel_size_5'],
                                 padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2_17']),dropout=space['l2_18'],recurrent_dropout=space['l2_19']))
            model.add(BatchNormalization())
            model.add(ConvLSTM2D(filters=space['filters_6'], kernel_size=space['kernel_size_6'],
                                 padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(space['l2_20']),dropout=space['l2_21'],recurrent_dropout=space['l2_22']))
            model.add(BatchNormalization())
        model.add(GlobalAveragePooling3D())
        model.add(Dense(space['Dense'],activation='relu'))
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
    lentruth=19600
# Train the network
    history = model.fit(
        train_generator,
        steps_per_epoch=lentrain/50.0,
        epochs=30,
        verbose=0,
        workers=0,
        use_multiprocessing=False,
        shuffle=True,validation_data=validation_generator,validation_steps=lentruth/50.0)
    score, acc=model.evaluate(validation_generator,steps=lentruth/50.0)
    modelnumber=next(tempfile._get_candidate_names())
    modelcode=np.random.randint(0,1e10)
    out = {'loss': -acc,
        'score': score,
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

def get_space():
    return {
        'filters': hp.choice('filters', [10,20,30,40,50,60]),
        'kernel_size': hp.choice('kernel_size', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2': hp.uniform('l2', 0,1),
        'l2_1': hp.uniform('l2_1', 0,1),
        'l2_2': hp.uniform('l2_2', 0,1),
        'filters_1': hp.choice('filters_1', [10,20,30,40,50,60]),
        'kernel_size_1': hp.choice('kernel_size_1', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2_3': hp.uniform('l2_3', 0,1),
        'l2_4': hp.uniform('l2_4', 0,1),
        'l2_5': hp.uniform('l2_5', 0,1),
        'filters_2': hp.choice('filters_2', [10,20,30,40,50,60]),
        'kernel_size_2': hp.choice('kernel_size_2', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2_6': hp.uniform('l2_6', 0,1),
        'l2_7': hp.uniform('l2_7', 0,1),
        'l2_8': hp.uniform('l2_8', 0,1),
        'filters_3': hp.choice('filters_3', [10,20,30,40,50,60]),
        'kernel_size_3': hp.choice('kernel_size_3', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2_9': hp.uniform('l2_9', 0,1),
        'l2_10': hp.uniform('l2_10', 0,1),
        'l2_11': hp.uniform('l2_11', 0,1),
        'l2_12': hp.choice('l2_12', ['four','five','six']),
        'filters_4': hp.choice('filters_4', [10,20,30,40,50,60]),
        'kernel_size_4': hp.choice('kernel_size_4', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2_13': hp.uniform('l2_13', 0,1),
        'l2_14': hp.uniform('l2_14', 0,1),
        'l2_15': hp.uniform('l2_15', 0,1),
        'l2_16': hp.choice('l2_16', ['four','five','six']),
        'filters_5': hp.choice('filters_5', [10,20,30,40,50,60]),
        'kernel_size_5': hp.choice('kernel_size_5', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2_17': hp.uniform('l2_17', 0,1),
        'l2_18': hp.uniform('l2_18', 0,1),
        'l2_19': hp.uniform('l2_19', 0,1),
        'filters_6': hp.choice('filters_6', [10,20,30,40,50,60]),
        'kernel_size_6': hp.choice('kernel_size_6', [(2,2),(3, 3),(4,4),(5,5),(6,6),(7,7)]),
        'l2_20': hp.uniform('l2_20', 0,1),
        'l2_21': hp.uniform('l2_21', 0,1),
        'l2_22': hp.uniform('l2_22', 0,1),
        'Dense': hp.choice('Dense', [10,50,100,200]),
    }
