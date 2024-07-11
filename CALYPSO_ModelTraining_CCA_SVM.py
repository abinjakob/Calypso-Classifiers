# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:07:22 2024

Classification of the SSVEP signal using SVM 
---------------------------------------------

All the data from the Calypso pilot data ws pooled together (6 datasets)

Feature used: CCA Correlation Values for stimulus frequencies and its harmonics
Classification: SVM classifier with 5-Fold crossvalidation
                - spliting data using train_test_split
                - scaling using StandarScalar
                - hyperparameter tuning using GridSearchCV

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de

@author: togo2120
"""

#%% libraries 

import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import glob
import mne 

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

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
    # raw.filter(2, 45, method= 'iir')
    
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
# chans2include = ['L2', 'L3', 'L4', 'R2', 'R3', 'R4']            # for 6 chan
chans2include = ['L2', 'R2']                                  # for 2 chan
# find indices of channels to include
chidx = [chnames.index(ch) for ch in chans2include if ch in chnames]

# make a copy eegdata 
eegdata_all = eegdata

# extract data from required channels 
eegdata = eegdata[:,chidx,:]
# check data shape 
print(eegdata.shape)

#%% computing CCA

# parameters for CCA
# number of epochs and samples 
numEpochs, _, tpts = eegdata.shape
# eeg data from the epocs 
eegEpoch = eegdata
# stimulation frequencies
freqs = [int(ev[0]), int(ev[1])]
# sampling frequency
fs = epochs.info["sfreq"]
# duration of epochs 
duration = tpts/fs
# generating time vector
t = np.linspace(0, duration, tpts, endpoint= False)

# initialising array to store features
CCAfeatures = []

# loop over epochs 
for iEpoch in range(numEpochs):
    # extract the X array
    X_data = eegEpoch[iEpoch,:,:].T
    # initialise array to store featues for each epoch
    epochFeat = []
    # loop over frequencies
    for i, iFreq in enumerate(freqs):    
        # create the sine and cosine signals for 1st harmonics
        sine1 = np.sin(2 * np.pi * iFreq * t)
        cos1 = np.cos(2 * np.pi * iFreq * t)
        # create the sine and cosine signals for 2nd harmonics
        sine2 = np.sin(2 * np.pi * (2 * iFreq) * t)
        cos2 = np.cos(2 * np.pi * (2 * iFreq) * t)
        
        # create Y vector 
        Y_data = np.column_stack((sine1, cos1, sine2, cos2))
        
        # performing CCA
        # considering the first canonical variables
        cca = CCA(n_components= 1)
        # compute cannonical variables
        cca.fit(X_data, Y_data)
        # return canonical variables
        Xc, Yc = cca.transform(X_data, Y_data)
        corr = np.corrcoef(Xc.T, Yc.T)[0,1]
        
        # store corr values for current epoch
        epochFeat.append(corr)
    
    # store features
    CCAfeatures.extend(epochFeat)
    
#%% Create feature and label vector

# feature vector (X)
X = np.array(CCAfeatures).reshape(numEpochs, -1)
# label vector (y)
y = labels 
    
#%% SVM classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define a pipeline with preprocessing (scaling) and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC())

# parameter grid for SVM
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # SVM regularization parameter
    'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
}

# apply cros-validaion on training set to find best SVM parameters
clf = GridSearchCV(pipeline, param_grid, cv=5)
# train the pipeline
clf.fit(X_train, y_train)

# display best parameters found by GridSearchCV
print(f'Best Parameters Found: {clf.best_params_}')

# make predictions
y_pred = clf.predict(X_test)

# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# calculate model performance
# accuracy
accuracy = accuracy_score(y_test, y_pred)
# precision (positive predictive value)
precision = precision_score(y_test, y_pred, labels=[1,2], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[1,2], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[1,2], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')
    
    
    
    
    
    
    
    
    
    
    
    

