# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:58:52 2024

Classification of the SSVEP signal using compact CNN (EEGNet) 
-------------------------------------------------------------

All the data from the Calypso pilot data ws pooled together (6 datasets)

Classification: Compact CNN 
                SSVEP version of the EEGNet was used for classification

EEGNet Model credits: 
The SSVEP version of EEGNet is based on Waytowich.et.al., 2018
http://stacks.iop.org/1741-2552/15/i=6/a=066031
Github: https://github.com/vlawhern/arl-eegmodels/tree/master


@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

#%% libraries 

import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import glob
import mne 

# EEGNet-specific imports
from EEGModels import EEGNet_SSVEP
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# explicitly set the tenserflow ordering to 'channels_last'
K.set_image_data_format('channels_last')

#%% read all pilot data from the folder 

# path to the file folder
folderpath = 'L:\Cloud\Calypso\Scripts\Pilot Data\Freqs_1520'
# read names of the files in the folder 
pilotFiles = glob.glob(op.join(folderpath, '*.set')) 

# empty lists to store all epochs, eeg data and labels
alldata = []
eegdata = []
labels  = []

# loop over fuile and read and epoch them in mne 
for file in pilotFiles:
    # load files to mne 
    raw = mne.io.read_raw_eeglab(file, eog= 'auto', preload= True)
    
    # drop channel R5 from all as in the last dataset this channel is not present
    if 'R5' in raw.info['ch_names']:
        raw.drop_channels(['R5'])
    
    # extract events
    events, eventinfo = mne.events_from_annotations(raw, verbose= False)
    # name of events
    ev = list(eventinfo.keys())
    # apply iir filter
    raw.filter(2, 45, method= 'iir')
    
    # epoch period 
    tmin, tmax = 0, 4
    # epoching 
    epochs = mne.Epochs(
        raw, 
        events= events,
        event_id= [eventinfo[ev[0]], eventinfo[ev[1]]],
        tmin= tmin, tmax= tmax, 
        baseline= None, 
        preload= True,
        event_repeated= 'merge',
        reject= {'eeg': 3.0})
    
    # save all data with epoch info
    alldata.append(epochs)
    # save just the data 
    eegdata.append(epochs.get_data())
    labels.append(epochs.events[:, -1])

# prepare the combined data and labels 
eegdata= np.concatenate(eegdata, axis=0)
labels = np.concatenate(labels, axis=0)

#%% check data shape
print(eegdata.shape)
print(labels.shape)

#%% select channels 

# all channel names 
chnames = epochs.info['ch_names']
# channels to include
chans2include = ['L2', 'L3', 'L4', 'R2', 'R3', 'R4']
# find indices of channels to include
chidx = [chnames.index(ch) for ch in chans2include if ch in chnames]

# make a copy eegdata 
eegdata_all = eegdata

# extract data from required channels 
eegdata = eegdata[:,chidx,:]
# check data shape 
print(eegdata.shape)

#%% create the data and label vector

# creating the data vector (trials x channels x samples)
# scaling by 1000 due to scaling sensitivity in deep learning 
X = eegdata * 1000
# creating the label vector 
y = labels
# define shape 
kernels = 1
chans   = X.shape[1]
samples = X.shape[2]

# split the data into train, validation and test set (50:25:25)
iDtrain = int((X.shape[0]*50)/100)
iDval   = int((X.shape[0]*25)/100)
iDtest  = int((X.shape[0]*25)/100)
X_train = X[0:iDtrain,]
y_train = y[0:iDtrain]
X_val   = X[iDtrain:iDtrain+iDval,]
y_val   = y[iDtrain:iDtrain+iDval]
X_test  = X[iDtrain+iDval:,]
y_test  = y[iDtrain+iDval:]

#%% prepare data and labels for EEGNet

# convert data to NHWC format (trials x channels x samples x kernels)
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_val   = X_val.reshape(X_val.shape[0], chans, samples, kernels)
X_test  = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# convert labels to one-hot encoding 
y_train = np_utils.to_categorical(y_train-1)
y_val   = np_utils.to_categorical(y_val-1)
y_test  = np_utils.to_categorical(y_test-1)

# final check on the data and label shape
print(f'Train Data shape: {X_train.shape}')
print(f'Train Label shape: {y_train.shape}')
print(f'Validation Data shape: {X_val.shape}')
print(f'Validation Label shape: {y_val.shape}')
print(f'Test Data shape: {X_train.shape}')
print(f'Test Label shape: {y_test.shape}')

#%% define the model

# number of classes
nb_classes = y_test.shape[1]
model = EEGNet_SSVEP(nb_classes= nb_classes, Chans= chans, Samples= samples, dropoutRate= 0.5, 
                     kernLength= 256, F1= 96, D= 1, F2= 96, dropoutType= 'Dropout')

# compile the model and set the optimizers
opt = Adam(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.compile(loss='binary_crossentropy', optimizer= opt, metrics = ['accuracy'])
# count number of parameters in the model
numParams = model.count_params()   
# print model summary
model.summary()
print(f'Parameters in the model: {numParams}')

# path to record model checkpoints 
# to save model weights at specific intervals during training 
checkpointer = ModelCheckpoint(filepath='L:\Cloud\Calypso\Scripts\Pilot Data\checkpoint.keras', verbose=1, save_best_only=True)

fittedModel = model.fit(X_train, y_train, batch_size = 16, epochs = 300, 
                        verbose = 2, validation_data=(X_val, y_val),
                        callbacks=[checkpointer])

#%% evaluate the model 

# load optimal saved weights after training 
model.load_weights('L:\Cloud\Calypso\Scripts\Pilot Data\checkpoint.keras')

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc   = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))





