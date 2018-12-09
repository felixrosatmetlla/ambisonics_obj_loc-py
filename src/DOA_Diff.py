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

mat_size = [np.shape(X_fq[2])[0],np.shape(X_fq[2])[1],3]

Xprime = np.empty(mat_size, dtype=np.complex128)
Xprime[:,:,0] = X_fq[2]
Xprime[:,:,1] = Y_fq[2]
Xprime[:,:,2] = Z_fq[2]

#%% Compute P and U vectors
ro = 1.204
c = 343
Zo = c*ro

#%% Compute the intensity vector and DOA

I = np.empty(mat_size, dtype=np.complex128)
I[:,:,0] = -1/(2*np.sqrt(2)*Zo)*(W_fq[2]*np.conj(Xprime[:,:,0]))
I[:,:,1] = -1/(2*np.sqrt(2)*Zo)*(W_fq[2]*np.conj(Xprime[:,:,1]))
I[:,:,2] = -1/(2*np.sqrt(2)*Zo)*(W_fq[2]*np.conj(Xprime[:,:,2]))

I = I.real

I_norm = np.linalg.norm(I, axis=2)

doa = np.empty(mat_size)
doa[:,:,0] = -np.nan_to_num(np.divide(I[:,:,0],I_norm))
doa[:,:,1] = -np.nan_to_num(np.divide(I[:,:,1],I_norm))
doa[:,:,2] = -np.nan_to_num(np.divide(I[:,:,2],I_norm))

#%%
norm_Xprime = np.empty(X_fq[2].shape, dtype=np.complex128)
norm_Xprime = np.linalg.norm(Xprime, axis=2) 
E = (np.power(norm_Xprime,norm_Xprime)/2 + np.power(np.absolute(W_fq[2]),np.absolute(W_fq[2])))

audio_data = np.empty(mat_size, dtype=np.complex128)
audio_data[:,:,0] = (W_fq[2]*np.conj(Xprime[:,:,0]))
audio_data[:,:,1] = (W_fq[2]*np.conj(Xprime[:,:,1]))
audio_data[:,:,2] = (W_fq[2]*np.conj(Xprime[:,:,2]))

diffueseness = np.empty(X_fq[2].shape, dtype=np.complex128)
#diffueseness = 1 - (np.sqrt(2)* )
for x in range(0,X_fq[2].shape[0]):
    for y in range(0,X_fq[2].shape[1]):
        if y < 9:
            avg_data = audio_data[x,0:y+1,:]
            avg_data2 = E[x,0:y+1]
            diffueseness[x,y] = 1 - (np.sqrt(2)* np.linalg.norm(np.average(avg_data)))/np.average(avg_data2)  
            
        else:
            avg_data = audio_data[x,y-9:y+1,:]
            avg_data2 = E[x,y-9:y+1]
            diffueseness[x,y] = 1 - (np.sqrt(2)* np.linalg.norm(np.average(avg_data)))/np.average(avg_data2) 
