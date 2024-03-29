'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.callbacks import ModelCheckpoint
import keras
import sys
from keras.models import load_model

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
from net_utils import *

plt.ioff()
runname=sys.argv[1]
modelfile='/users/exet4487/hypermodels/'+str(sys.argv[1])+'.h5'

oldmodel=load_model(modelfile)

config=oldmodel.get_config()
print(config)



# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
hexmethod='oversampling'

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



# Train the network

checkpoint_filepath = '/mnt/extraspace/exet4487/'+runname+'_checkpoint/'
try:
    os.system('mkdir '+checkpoint_filepath)
except:
    print('Folder already exists')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath+'model.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='auto',
    save_best_only=False)
strategy=tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model=keras.Sequential.from_config(config)
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['binary_accuracy'])
    print(model.summary())
history = model.fit_generator(
    generate_training_sequences(onlyfiles,
        50,
                                'Train',hexmethod),
    steps_per_epoch=lentrain/50.0,
    epochs=100,
    verbose=2,
    workers=0,callbacks=[model_checkpoint_callback],
    use_multiprocessing=False,
    shuffle=True,validation_data=generate_training_sequences(onlyfiles,50,'Valid',hexmethod),validation_steps=lentruth/50.0)

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
pred = model.predict_generator(
    generate_training_sequences(onlyfiles,
        1,
        'Test',hexmethod),
    verbose=0,workers=0,
     use_multiprocessing=False,
    steps=len(Trutharr))
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
