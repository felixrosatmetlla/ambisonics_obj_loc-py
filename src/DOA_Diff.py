#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:41:24 2018

@author: felixrosatmetlla
"""

#Import the needed libraries to read audios, use numpy arrays and plot
import soundfile as sf
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import os

#%%

def getBFormatAudioPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test_audios/output/'+ output_filename) 
    return output_path


#%% Get Path and read audio file
    
bformat_pth = getBFormatAudioPath('violinsingle_FUMA_FUMA(135, 0).wav')

#Read audio file
data, samplerate = sf.read(bformat_pth)

#We get each channel individually
W = data[:,0]
X = data[:,1]
Y = data[:,2]
Z = data[:,3]

#%% We use STFT to get frequency domain of each channel

W_fq = sig.stft(W, samplerate, 'hann', 256)
X_fq = sig.stft(X, samplerate, 'hann', 256)
Y_fq = sig.stft(Y, samplerate, 'hann', 256)
Z_fq = sig.stft(Z, samplerate, 'hann', 256)

#%% Compute P and U vectors
Zo = 343*1.204

P = W_fq[2]
Ux = 1/(np.sqrt(2))*X_fq[2]
Uy = 1/(np.sqrt(2))*Y_fq[2]
Uz = 1/(np.sqrt(2))*Z_fq[2]
