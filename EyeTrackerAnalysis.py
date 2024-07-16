#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:56:37 2024

Analysis Eye Tracker Data saved from Hololens 
---------------------------------------------

Loads the Eye Tracker CSV file and analyses the X and Y coordinates of the data 
and plots them in reference to the flicker stim. 

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

#%% libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
import matplotlib.patches as patches

#%% open the csv as data frames 

# path to file 
rootpath = r'/Users/abinjacob/Documents/01. Calypso/Calpso 1.0/Eye Tracker Data'
filename = 'eyetrackerdata_hololens_test5_16072024.csv'
filepath = op.join(rootpath,filename)

# assigning column names 
clnames = ['time', 'xdim', 'ydim', 'zdim']
# read csv file into df
df = pd.read_csv(filepath, header= None, names= clnames)

#%% plotting data 

plt.scatter(df['xdim'], df['ydim'], color= 'red', alpha= 0.1, s= 10)
square = patches.Rectangle((0-0.110, 0-0.110), 0.220, 0.220, linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(square)
plt.xlim(-.7, .7)
plt.ylim(-.7, .7)
plt.title('Eye Tracker Data')



