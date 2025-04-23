#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:33:27 2025

CalypsoFLICK algorithm - v2
---------------------------

This code is modified from CalypsoFLICK_V1
The code loads SSVEP EEG data and Eye-Tracker data while performing SSVEP experiment in AR.
Both the EEG and Eye-Tracker data from left and right trials are divided into fixed length 
epochs (eg. 4 seconds).  

The SSVEP data is classified using CCA-SVM algorithm and the propbabilities for each class
are determined. The Eye-tracking data is classified using Nearest-Neighbor method and the 
probabilities of the classes are determined. A Bayesian Fusion Approach is then implemented
to determine the final class of each trial.

@author: abinjacob
"""

# libraries 

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

# load data 
rootpath = r'/Users/abinjacob/Documents/01. Calypso/Calpso 1.0/Scripts/Classification in Python/CalypsoFLICK_Data'
# eegfiles to load 
eegfiles = ['Pilot13_SSVEPleft_4LeftChan.set', 'Pilot13_SSVEPright_4LeftChan.set']
# eye-tracker files to load 
eyefiles = ['pilot13_eyetracker_left.csv', 'pilot13_eyetracker_right.csv']

# params 
# duration of epochs (in sec)
epoch_duration = 4
# stimulation frequencies
freqs = [15, 20]
# sampling rate of eye-tracker
ETsrate = 60
# spatial coordinates of the left and right flicker syimulus center
leftCenter = np.array([-0.2, 0])   
rightCenter = np.array([0.2, 0])  

# eyeConfidence = []

#%% CLASSIFYING SSVEP DATA

# variable inits 
Xeeg_all  = []
labels = []

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
    ccafeat = np.array(CCAfeatures).reshape(numEpochs, -1)
    Xeeg_all.append(ccafeat)
    # temporary label vector
    labels.extend([iFile + 1] * numEpochs)

# Create feature vector for SSVEP data
Xeeg = np.vstack(Xeeg_all)
# Create label vector for left and right attended trials
y = np.array(labels)

# plotting the feature space of SSVEP data
plt.figure()
plt.scatter(Xeeg[:,0][y==1], Xeeg[:,1][y==1], label='Class 1')
plt.scatter(Xeeg[:,0][y==2], Xeeg[:,1][y==2], label='Class 2')
plt.title('SSVEP Feature Space')

# SVM classifier with 5 fold cross-validation for SSVEP data

# split the dataset into trainning and testing set
indices = np.arange(len(Xeeg))
Xeeg_train, Xeeg_test, y_train, y_test, train_idx, test_idx = train_test_split(Xeeg, y, indices, test_size=0.3, random_state=42)

# define a pipeline with preprocessing (scaling) and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC(probability=True))

# parameter grid for SVM
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # SVM regularization parameter
    'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
}

# apply cros-validaion on training set to find best SVM parameters
clf = GridSearchCV(pipeline, param_grid, cv=5)
# train the pipeline
clf.fit(Xeeg_train, y_train)

# display best parameters found by GridSearchCV
print(f'Best Parameters Found: {clf.best_params_}')

# make predictions
y_pred = clf.predict(Xeeg_test)
P_eeg = clf.predict_proba(Xeeg_test)

# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# calculate model performance
# accuracy
accuracy_eeg = accuracy_score(y_test, y_pred)
# precision (positive predictive value)
precision = precision_score(y_test, y_pred, labels=[1,2], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[1,2], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[1,2], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('SSVEP Model Performance Metrics')
print(f'SSVEP Classification Accuracy: {accuracy_eeg*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%')

# plotting the SVM performance with SSVEP data
plt.figure()
plt.scatter(Xeeg_test[:,0][y_test==1], Xeeg_test[:,1][y_test==1], label='Class 1')
plt.scatter(Xeeg_test[:,0][y_test==2], Xeeg_test[:,1][y_test==2], label='Class 2')
plt.scatter(Xeeg_test[:,0][y_pred==1], Xeeg_test[:,1][y_pred==1], label='Pred Class 1', marker= 'o', facecolors= 'none', edgecolors='blue', linewidth=1)
plt.scatter(Xeeg_test[:,0][y_pred==2], Xeeg_test[:,1][y_pred==2], label='Pred Class 2', marker= 'o', facecolors= 'none', edgecolors='red', linewidth=1)
plt.xlabel('cca coeff for class 1')
plt.ylabel('cca coeff for class 2')
plt.title(f'SVM Model Prediction for SSVEP Data (Acc: {accuracy_eeg*100:.2f}%)')
plt.legend()

#%% CLASSIFYING EYE-TRACKING DATA

# variable inits 
eyedata = []
P_eye = []
y_predeye = []

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
    eyedata.append(eye_temp)

# eye tracker data
eyedata = np.vstack(eyedata)
Xeye = eyedata[test_idx]


# plotting eye data 
# plotting the left and right condition eye tracker data
plt.figure()
plt.scatter(eyedata[:,:,0][y==1], eyedata[:,:,1][y==1], color= 'red', alpha= 0.1, s= 10, label= 'Class 1')
plt.scatter(eyedata[:,:,0][y==2], eyedata[:,:,1][y==2], color= 'blue', alpha= 0.1, s= 10, label= 'Class 2')
square15 = patches.Rectangle((0-0.29, 0-0.08), 0.15, 0.17, linewidth=1, edgecolor='r', facecolor='none')
square20 = patches.Rectangle((0+0.12, 0-0.08), 0.15, 0.17, linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(square15)
plt.gca().add_patch(square20)
plt.xlim(-.4, .4)
plt.ylim(-.4, .4)
plt.title('Eye Tracker Data')
plt.legend()


# Initialize probabilities list
ETtrialprobs = []

for iTrial in range(Xeye.shape[0]):
    # extract the X, Y positions for the current trial
    ETtrialdata = Xeye[iTrial, :, :]
    
    # calculate distances to the left and right boxes for each time point
    distancesLeft = cdist(ETtrialdata, [leftCenter], metric='euclidean').flatten()
    distancesRight = cdist(ETtrialdata, [rightCenter], metric='euclidean').flatten()
    
    # converting distances to probabilities using softmax
    # leftProb = np.exp(-distancesLeft) / (np.exp(-distancesLeft) + np.exp(-distancesRight) + 1e-10)
    # rightProb = np.exp(-distancesRight) / (np.exp(-distancesLeft) + np.exp(-distancesRight) + 1e-10)
    
    # converting distances to probabilities
    P_eye_class1 = distancesRight / (distancesLeft + distancesRight + 1e-10)
    P_eye_class2 = distancesLeft / (distancesLeft + distancesRight + 1e-10)

    # average the probabilities within the epoch
    P_eye_class1_avg = np.mean(P_eye_class1)
    P_eye_class2_avg = np.mean(P_eye_class2)
    
    # store the averaged probabilities
    P_eye.append([P_eye_class1_avg, P_eye_class2_avg])
    
    # classifying eye-data
    if P_eye_class1_avg > P_eye_class2_avg:
        y_predeye.append(1)
    else:
        y_predeye.append(2)
        
# convert to numpy array for further processing
y_predeye = np.array(y_predeye)
P_eye = np.array(P_eye)
accuracy_eye = np.mean(y_predeye == y_test)
print('Eye-Tracking Model Performance Metrics')
print(f'Eye-Tracker Classification Accuracy: {accuracy_eye*100:.2f}%')

# plotting eye-tracker prediction results
plt.figure()
plt.scatter(np.mean(Xeye[:,:,0][y_test==1], axis=1), np.mean(Xeye[:,:,1][y_test==1], axis=1), label= 'Class 1')
plt.scatter(np.mean(Xeye[:,:,0][y_test==2], axis=1), np.mean(Xeye[:,:,1][y_test==2], axis=1), label= 'Class 2')
plt.scatter(np.mean(Xeye[:,:,0][y_predeye==1], axis=1), np.mean(Xeye[:,:,1][y_predeye==1], axis=1), label='Pred Class 1', marker= 'o', facecolors= 'none', edgecolors='blue', linewidth=1)
plt.scatter(np.mean(Xeye[:,:,0][y_predeye==2], axis=1), np.mean(Xeye[:,:,1][y_predeye==2], axis=1), label='Pred Class 2', marker= 'o', facecolors= 'none', edgecolors='red', linewidth=1)
plt.title(f'Prediction for Eye-Tracker Data (Acc: {accuracy_eye*100:.2f}%)')
plt.legend()

#%% DECISION FUSION - Based on Bayesian Fusion Approach

# variable init
y_predfusion = []

# prior probabilities for each class (equal prior for Left and Right)
P_prior = np.array([0.5, 0.5])  

# function for Bayesian Fusion
def bayesian_fusion(P_eeg, P_eye, P_prior):
    # joint probability
    P_joint = P_eeg * P_eye * P_prior  
    # normalize to get posterior probabilities
    P_fusion = np.divide(P_joint, np.sum(P_joint, axis=1, keepdims=True), 
                         where=np.sum(P_joint, axis=1, keepdims=True) != 0)  
    return np.argmax(P_fusion, axis=1) + 1

# calculate predictions using Bayesian fusion
y_predfusion = bayesian_fusion(P_eeg, P_eye, P_prior)

# compute accuracy
accuracy_fusion = np.mean(y_predfusion == y_test)
print('Model Comparison')
print(f'SSVEP Classification Accuracy: {accuracy_eeg*100:.2f}%')
print(f'Eye-Tracker Classification Accuracy: {accuracy_eye*100:.2f}%')
print(f"CalypsoFLICK Accuracy: {accuracy_fusion * 100:.2f}%")


# plotting CalypsoFLICK prediction results
plt.figure()
plt.scatter(Xeeg_test[:,0][y_test==1], Xeeg_test[:,1][y_test==1], label='Class 1')
plt.scatter(Xeeg_test[:,0][y_test==2], Xeeg_test[:,1][y_test==2], label='Class 2')
plt.scatter(Xeeg_test[:,0][y_predfusion==1], Xeeg_test[:,1][y_predfusion==1], label='Pred Class 1', marker= 'o', facecolors= 'none', edgecolors='blue', linewidth=1)
plt.scatter(Xeeg_test[:,0][y_predfusion==2], Xeeg_test[:,1][y_predfusion==2], label='Pred Class 2', marker= 'o', facecolors= 'none', edgecolors='red', linewidth=1)
plt.xlabel('cca coeff for class 1')
plt.ylabel('cca coeff for class 2')
plt.title(f'CalypsoFLICK Model Prediction (Acc: {accuracy_fusion*100:.2f}%)')
plt.legend()


