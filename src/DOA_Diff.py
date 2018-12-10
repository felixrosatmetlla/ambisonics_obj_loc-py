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

#%% Functions

def getBFormatAudioPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test_audios/output/'+ output_filename) 
    return output_path

def getFreqDomain(W,X,Y,Z):
    W_fq = sig.stft(W, samplerate, 'hann', 256)
    X_fq = sig.stft(X, samplerate, 'hann', 256)
    Y_fq = sig.stft(Y, samplerate, 'hann', 256)
    Z_fq = sig.stft(Z, samplerate, 'hann', 256)
    
    return W_fq[2], X_fq[2], Y_fq[2], Z_fq[2]

def getXprime(X_fq, Y_fq, Z_fq):
    Xprime_Size = [np.shape(X_fq)[0],np.shape(X_fq)[1], 3]
    
    Xprime = np.empty(Xprime_Size, dtype=np.complex128)
    Xprime[:,:,0] = X_fq
    Xprime[:,:,1] = Y_fq
    Xprime[:,:,2] = Z_fq
    
    return Xprime, Xprime_Size

def getIntVec(W_fq, Xprime, Int_Size, Zo):
    I = np.empty(Int_Size, dtype=np.complex128)
    I[:,:,0] = -1/(2*np.sqrt(2)*Zo)*(W_fq[2]*np.conj(Xprime[:,:,0]))
    I[:,:,1] = -1/(2*np.sqrt(2)*Zo)*(W_fq[2]*np.conj(Xprime[:,:,1]))
    I[:,:,2] = -1/(2*np.sqrt(2)*Zo)*(W_fq[2]*np.conj(Xprime[:,:,2]))    
    I = I.real
    
    return I

def DOA(I, doa_Size):
    I_norm = np.linalg.norm(I, axis=2)
    
    doa = np.empty(doa_Size)
    doa[:,:,0] = -np.nan_to_num(np.divide(I[:,:,0],I_norm))
    doa[:,:,1] = -np.nan_to_num(np.divide(I[:,:,1],I_norm))
    doa[:,:,2] = -np.nan_to_num(np.divide(I[:,:,2],I_norm))
    
    hxy = np.hypot(doa[:,:,0], doa[:,:,1])
    r = np.hypot(hxy, doa[:,:,2])
    el = np.arctan2(doa[:,:,2], hxy)
    az = np.arctan2(doa[:,:,1], doa[:,:,0])
    
    return doa, r, el, az

def Diffuseness(Xprime, W_fq ):
    norm_Xprime = np.empty(W_fq.shape, dtype=np.complex128)
    norm_Xprime = np.linalg.norm(Xprime, axis=2) 
    E = (np.power(norm_Xprime,norm_Xprime)/2 + np.power(np.absolute(W_fq),np.absolute(W_fq)))
    
    audio_data = np.empty(Xprime_Size, dtype=np.complex128)
    audio_data[:,:,0] = (W_fq*np.conj(Xprime[:,:,0]))
    audio_data[:,:,1] = (W_fq*np.conj(Xprime[:,:,1]))
    audio_data[:,:,2] = (W_fq*np.conj(Xprime[:,:,2]))
    
    audio_data = audio_data.real
    
    diffueseness = np.empty(W_fq.shape)
    #diffueseness = 1 - (np.sqrt(2)* )
    for x in range(0,W_fq.shape[0]):
        for y in range(0,W_fq.shape[1]):
            if y < 9:
                avg_data = audio_data[x,0:y+1,:]
                avg_data2 = E[x,0:y+1]
                diffueseness[x,y] = 1 - (np.sqrt(2)* np.linalg.norm(np.average(avg_data)))/np.average(avg_data2)  
                
            else:
                avg_data = audio_data[x,y-9:y+1,:]
                avg_data2 = E[x,y-9:y+1]
                diffueseness[x,y] = 1 - (np.sqrt(2)* np.linalg.norm(np.average(avg_data)))/np.average(avg_data2)
                
    return diffueseness
#%% Get Path and read audio file
    
bformat_pth = getBFormatAudioPath('violinsingle_FUMA_FUMA(0, 0).wav')

#Read audio file
data, samplerate = sf.read(bformat_pth)

#We get each channel individually
W = data[:,0]
X = data[:,1]
Y = data[:,2]
Z = data[:,3]

#%% We use STFT to get frequency domain of each channel

W_fq, X_fq, Y_fq, Z_fq = getFreqDomain(W,X,Y,Z)

Xprime, Xprime_Size = getXprime(X_fq, Y_fq, Z_fq)

#%% Compute the intensity vector and DOA
ro = 1.204
c = 343
Zo = c*ro

I = getIntVec(W_fq, Xprime, Xprime_Size, Zo)

doa, r, el, az = DOA(I, Xprime_Size)

#%% Diffuseness computation

diffueseness = Diffuseness(Xprime, W_fq )

#%% Plots
            
plt.specgram(diffueseness,256,samplerate,900)
plt.show()
