'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import mnist
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

tf.compat.v1.disable_eager_execution()

# Parameters
l2_reg = 5e-4         # Regularization rate for l2
learning_rate = 1e-3  # Learning rate for SGD
batch_size = 32       # Batch size
epochs = 1000         # Number of training epochs
es_patience = 10      # Patience fot early stopping
plt.ioff()

# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/Crab64080/*.hdf5'))
runname = 'graph1'
hexmethod='oversampling'
n_out = 10                 # Dimension of the target

fltr = GraphConv.preprocess(A)

global Trutharr
Trutharr = []
Train2=[]
truid=[]
print(onlyfiles,len(onlyfiles))

# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[120:160]:
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

for file in onlyfiles[:120]:
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

model = Sequential()
model.add(ConvLSTM2D(filters=10, kernel_size=(4, 4),
                     input_shape=inpshape,
                     padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.0797),dropout=0.0361,recurrent_dropout=0.382))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=10, kernel_size=(2, 2),
                     padding='same', return_sequences=True,dropout=0.371,recurrent_dropout=0.510,kernel_regularizer=keras.regularizers.l2(0.576)))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(2, 2),
                     padding='same', return_sequences=True,dropout=0.0))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),
                     padding='same', return_sequences=True,dropout=0.5056))
model.add(BatchNormalization())
model.add(GlobalAveragePooling3D())
model.add(Dense(100,activation='relu'))
model.add(Dense(2, activation='softmax'))
opt = keras.optimizers.Adam(lr=1e-5)

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['binary_accuracy'])

'''early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')'''

# Code for ensuring no contamination between training and test data.
print(model.summary())

plot_model(
    model,
    to_file='/users/exet4487/Figures/'+runname+'_model.png',
    show_shapes=True)

# Train the network
history = model.fit_generator(
    generate_training_sequences(onlyfiles,
        20,
                                'Train',hexmethod),
    steps_per_epoch=lentrain/20.0,
    epochs=50,
    verbose=2,
    workers=0,
    use_multiprocessing=False,
    shuffle=True,validation_data=generate_training_sequences(onlyfiles,20,'Valid',hexmethod),validation_steps=lentruth/20.0)

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

score = model.evaluate_generator(generate_training_sequences(onlyfiles,20,'Test',hexmethod),workers=0,use_multiprocessing=False,steps=len(Trutharr)/20.0)
model.save('/users/exet4487/Models/'+runname+'model.hdf5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix


print(get_confusion_matrix_one_hot(runname,pred, Trutharr))
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
cutval=0.1
print('No_gamma',len(np.where(p2>=cutval)[0]))
print('No_bg',len(np.where(p2<cutval)[0]))
