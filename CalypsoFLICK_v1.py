#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:45:42 2024

CalypsoFLICK algorithm - v1
---------------------------

This is the initial algorithm for the classification of SSVEP signals and Eye-tracker
data collected while performing the CalypsoFLICK AR Application. CalypsoFLICK AR app
in its current state displays 2 flicker boxes adjacent to each other. However, as the 
LSL communication is not yet implemented, the flickers are presented continuously for 
120 sec for left and right trials. The participant attend for 120sec to the left stim
and then to the right stim. The EEG signals are recorded using SMARTING to a computer
and the eyetracker data is recorded using the in-built eye-tracker of the hololens which
is then saved to the App Files of the hololens. 

Both the EEG and Eye-Tracker data from left and right trials are divided into fixed length 
epochs (eg. 4 seconds) for further classification procedures. 

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

#%% libraries 

import mne
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy.spatial.distance import cdist

#%% setups 

rootpath = r'/Users/abinjacob/Documents/01. Calypso/Calpso 1.0/Scripts/Classification in Python/CalypsoFLICK_Data'
# eegfiles to load 
eegfiles = ['Pilot13_SSVEPleft_4LeftChan.set', 'Pilot13_SSVEPright_4LeftChan.set']
# eye-tracker files to load 
eyefiles = ['pilot13_eyetracker_left.csv', 'pilot13_eyetracker_right.csv']

# duration of epochs (in sec)
epoch_duration = 4

# stimulation frequencies
freqs = [15, 20]

# sampling rate of eye-tracker
ETsrate = 60

# center coordinates of the left and right box 
leftCenter = np.array([-0.2, 0])   
rightCenter = np.array([0.2, 0])  

# matrix initialisation
X_all  = []
labels = []
eye_all = []
ETtrialprobs = []
y_predeye = []
eyeConfidence = []

#%% CLASSIFYING SSVEP DATA

# loading the SSVEP EEG data 
# Note: 2 files has to be loaded as LSL was not implemented and hence 
# left and right attended trials were collected seperately as 2 blocks
for iFile in range(len(eegfiles)):  
    # EEGLab file to load (.set)
    filename = eegfiles[iFile]
    # filename = 'SSVEP_SHIT.set'
    filepath = op.join(rootpath,filename)
    # load file in mne 
    raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)
    
    # cropping the first and last 10sec of data to avoid non-SSVEP EEG data 
    raw.crop(tmin=10, tmax= (raw.times[-1] - 10))
    
    # epoching the data with fixed length
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
    print(epochs.get_data().shape)
    
    # parameters for CCA
    # number of epochs and samples 
    numEpochs, _, tpts = epochs.get_data().shape
    # eeg data from the epocs 
    eegEpoch = epochs.get_data()
    
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
            # Y_data = np.column_stack((sine1, cos1, sine2, cos2))
            Y_data = np.column_stack((sine1, cos1))  # *****.*****.****.****.****.***** ONLY FIRST HARMONICS INCLUDED
            
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
    
    # temporary feature vector (X)
    X_temp = np.array(CCAfeatures).reshape(numEpochs, -1)
    X_all.append(X_temp)
    # temporary label vector
    labels.extend([iFile + 1] * numEpochs)

# Create feature vector for SSVEP data
X = np.vstack(X_all)
# Create label vector for left and right attended trials
y = np.array(labels)

# plotting the feature space of SSVEP data
plt.figure()
plt.scatter(X[:,0][y==1], X[:,1][y==1], label='label 1')
plt.scatter(X[:,0][y==2], X[:,1][y==2], label='label 2')
plt.title('SSVEP Feature Space')

# SVM classifier with 5 fold cross-validation for SSVEP data

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

# plotting the SVM performance with SSVEP data
plt.figure()
plt.scatter(X_test[:,0][y_test==1], X_test[:,1][y_test==1], label='label 1')
plt.scatter(X_test[:,0][y_test==2], X_test[:,1][y_test==2], label='label 2')
plt.scatter(X_test[:,0][y_pred==1], X_test[:,1][y_pred==1], label='pred 1', marker= 'o', facecolors= 'none', edgecolors='blue', linewidth=1)
plt.scatter(X_test[:,0][y_pred==2], X_test[:,1][y_pred==2], label='pred 2', marker= 'o', facecolors= 'none', edgecolors='red', linewidth=1)
plt.xlabel('cca coeff for class 1')
plt.ylabel('cca coeff for class 2')
plt.title(f'SVM Model Prediction for SSVEP Data (Acc: {accuracy*100:.2f}%)')
plt.legend()

#%% CLASSIFYING EYE-TRACKER DATA

for iFile in range(len(eyefiles)): 
    # loading 
    filename = eyefiles[iFile]
    filepath = op.join(rootpath,filename)
    
    # assigning column names 
    clnames = ['time', 'xdim', 'ydim', 'zdim']
    # read csv file into df
    df = pd.read_csv(filepath, header= None, names= clnames)
    
    # determining eye-tracker samples in an epoch
    ETepochsamples = ETsrate * epoch_duration
    ETnepochs = df.shape[0] // ETepochsamples
    
    # reshape the data into epochs 
    ETepochs = df.iloc[:ETnepochs * ETepochsamples].values.reshape(ETnepochs, ETepochsamples, df.shape[1])
    # choosing only the X and Y dimension values with trial period
    eye_temp = ETepochs[2:32, :, 1:3]
    eye_all.append(eye_temp)

# eye tracker data
X_eye = np.vstack(eye_all)

# plotting the left and right condition eye tracker data
plt.figure()
plt.scatter(X_eye[:,:,0][y==1], X_eye[:,:,1][y==1], color= 'red', alpha= 0.1, s= 10, label= 'label 1')
plt.scatter(X_eye[:,:,0][y==2], X_eye[:,:,1][y==2], color= 'blue', alpha= 0.1, s= 10, label= 'label 2')
square15 = patches.Rectangle((0-0.29, 0-0.08), 0.15, 0.17, linewidth=1, edgecolor='r', facecolor='none')
square20 = patches.Rectangle((0+0.12, 0-0.08), 0.15, 0.17, linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(square15)
plt.gca().add_patch(square20)
plt.xlim(-.4, .4)
plt.ylim(-.4, .4)
plt.title('Eye Tracker Data')
plt.legend()


# classifying the eye-tracker data based on nearest neighbours 
# loop over each trial
for iTrial in range(X_eye.shape[0]):
    
    # extract the X, Y positions for the current trial 
    ETtrialdata = X_eye[iTrial, :, :]
    
    # calculate the distances to the left and right boxes for each time point
    distancesLeft = cdist(ETtrialdata, [leftCenter], metric='euclidean').flatten()
    distancesRight = cdist(ETtrialdata, [rightCenter], metric='euclidean').flatten()
    
    # converting distances to probabilities
    leftProb = distancesRight / (distancesLeft + distancesRight + 1e-10)
    rightProb = distancesLeft / (distancesLeft + distancesRight + 1e-10)

    # average the probabilities within epoch
    leftProbAvg = np.mean(leftProb)
    rightProbAvg = np.mean(rightProb)
    
    # store the averaged probability
    ETtrialprobs.append([leftProbAvg, rightProbAvg])
    
    
    # classification logic
    # classify left:
    if leftProbAvg >= 0.9 and rightProbAvg <=0.1:
        y_predeye.append(1)
        eyeConfidence.append(100)
    elif leftProbAvg >= 0.7 and rightProbAvg <=0.3:
        y_predeye.append(1)
        eyeConfidence.append(80)
    elif leftProbAvg >= 0.5 and rightProbAvg <0.5:
        y_predeye.append(1)
        eyeConfidence.append(50)
    
    # classify right:
    elif rightProbAvg >= 0.9 and leftProbAvg <=0.1:
        y_predeye.append(2)
        eyeConfidence.append(100)
    elif rightProbAvg >= 0.7 and leftProbAvg <=0.3:
        y_predeye.append(2)
        eyeConfidence.append(80)
    elif rightProbAvg >= 0.5 and leftProbAvg <0.5:
        y_predeye.append(2)
        eyeConfidence.append(50)
    else: 
        y_predeye.append(0)
        eyeConfidence.append(0)
        
# convert the list to a numpy array
ETtrialprobs = np.array(ETtrialprobs)
# calculate accuracy
ETacc = np.mean(y_predeye == y)
ETconf = np.mean(eyeConfidence)

# Print the accuracy
print(f"Accuracy: {ETacc * 100:.2f}%")
print(f"Confidence: {ETconf:.2f}%")




