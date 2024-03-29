'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.callbacks import ModelCheckpoint
import keras
import sys
from keras.models import load_model
import random
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
from net_utils import get_confusion_matrix_one_hot
from numpy import moveaxis
import tqdm
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras import backend as K
plt.ioff()
runname=sys.argv[1]



# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
hexmethod='oversampling'
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
global Trutharr
Trutharr = []
Train2=[]
truid=[]
print(onlyfiles,len(onlyfiles))

# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[-90:-50]:
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        print('File failed to load',file)
        continue
    labelsarr = np.asarray(inputdata['isGamma'][:])
    idarr = np.asarray(inputdata['id'][:])
    for value in labelsarr:
        Trutharr.append(value)
    for value in idarr:
        truid.append(value)
    inputdata.close()

for file in onlyfiles[20:-90]:
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        print('File failed to load',file)
        continue
    labelsarr = np.asarray(inputdata['isGamma'][:])
    for value in labelsarr:
        Train2.append(value)
    inputdata.close()

print('lentruth', len(Trutharr))
print('lentrain',len(Train2))

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
        filelist = onlyfiles[20:-90]
        print('train', filelist)
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                print('File failed to load',file)
                continue
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[-90:-50]
        print('test', filelist)
        global testevents
        global test2
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                print('File failed to load',file)
                continue
            testevents = testevents + inputdata['isGamma'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Valid':
        filelist = onlyfiles[-50:-10]
        print('valid', filelist)
        global validevents
        global valid2
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                print('File failed to load',file)
                continue
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
                X = X[:,:,:,:,0]
                X=moveaxis(X,1,3)
                yield (np.array(X), np.array(Y))

lentrain=len(Train2)
lentruth=len(Trutharr)
np.save('/users/exet4487/truesim/truthvals_'+runname+'.npy',np.asarray(Trutharr))
np.save('/users/exet4487/idsim/idvals_'+runname+'.npy',np.asarray(truid))

# Define model architecture.
if hexmethod in ['axial_addressing','image_shifting']:
    inpshape=(None,27,27,1)
elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversampling','rebinning']:
    inpshape=(None,54,54,1)
else:
    print('Invalid Hexmethod')
    raise KeyboardInterrupt

# Code for ensuring no contamination between training and test data.


def get_model():
    
    inp = Input(shape=(54,54,4))

    x = Conv2D(32, (3,3), activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x, training = True)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x, training = True)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x, training = True)

    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

print(np.shape(next(generate_training_sequences(onlyfiles,50,'Train',hexmethod))[0]))

model=get_model()
print(model.summary())
tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=7)

history = model.fit_generator(generate_training_sequences(onlyfiles,50,'Train',hexmethod),steps_per_epoch=lentrain/50.0,
                              epochs=10,verbose=2,workers=0,use_multiprocessing=False,shuffle=False,
                              callbacks=[es],validation_data=generate_training_sequences(onlyfiles,50,'Valid',hexmethod),validation_steps=lentruth/50.0)

# Plot training accuracy/loss.
fig = plt.figure()
plt.subplot(2, 1, 1)
print(history.history)
plt.plot(history.history['binary_accuracy'], label='Train')
plt.plot(history.history['val_binary_accuracy'],label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'],label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig('/users/exet4487/Figures/'+runname+'trainlog.png')

# Test the network
print('Predicting')

### REACTIVATE DROPOUTS AND ITERATE PREDICTIONS ###

pred_bayes_dist = []
rep = 100

for i in tqdm(range(0,rep)):
    pred_bayes_dist.append(pred = model.predict_generator(
    generate_training_sequences(onlyfiles,
        1,
        'Test',hexmethod),
    verbose=0,workers=0,
     use_multiprocessing=False,
    steps=len(Trutharr))
)
    
pred_bayes_dist = np.transpose(np.stack(pred_bayes_dist), (1,0,2))
print(pred_bayes_dist.shape)
raise KeyboardInterrupt
np.save('/users/exet4487/predictions/'+runname+'_predictions.npy', pred)

print('Evaluating')

score = model.evaluate_generator(generate_training_sequences(onlyfiles,50,'Test',hexmethod),workers=0,use_multiprocessing=False,steps=len(Trutharr)/50.0)
model.save('/users/exet4487/Models/'+runname+'model.hdf5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix


print(get_confusion_matrix_one_hot(runname,'/users/exet4487',pred, Trutharr))
fig=plt.figure()
Trutharr=np.asarray(Trutharr)
noev=min([len(Trutharr),len(pred)])
pred=pred[:noev]
Trutharr=Trutharr[:noev]
x1=np.where(Trutharr==0)
x2=np.where(Trutharr==1)
p2=[]
print(pred,np.shape(pred))

for i in np.arange(np.shape(pred)[0]):
    score=np.argmax(pred[i])
    if score==0:
        s2=1-pred[i][0]
    elif score==1:
        s2=pred[i][1]
    p2.append(s2)


p2=np.asarray(p2)
np.save('/users/exet4487/predictions/'+runname+'_predictions.npy', p2)
x1=x1[0]
x2=x2[0]

plt.hist(p2[x1],10,label='True Hadron',alpha=0.5,density=False)
plt.hist(p2[x2],10,label='True Gamma',alpha=0.5,density=False)
plt.xlabel('isGamma Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('/users/exet4487/Figures/'+runname+'_hist.png')
cutval=0.35
print('No_gamma',len(np.where(p2>=cutval)[0]))
print('No_bg',len(np.where(p2<cutval)[0]))
