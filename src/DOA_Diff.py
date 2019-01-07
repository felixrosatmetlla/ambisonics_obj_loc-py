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
    output_path = os.path.join(output_path,'test/output/'+ output_filename) 
    return output_path

def plotSignal(title, x):
    plt.figure()
    plt.suptitle(title)
    plt.plot(x)
    
def plotSpectrogram(title, x_fq, colorMap):
    plt.figure()
    plt.suptitle(title)
    plt.pcolormesh(x_fq, cmap = colorMap)
    plt.colorbar()
    
def to_dB(x, intensity):
    signal = np.abs(x.real) + 1e-10
    if intensity == 'Y':
        x_db = 20*np.log(signal)
    elif intensity == 'N':
        x_db = 10*np.log(signal)
    
    return x_db

def cart2Sph(x):
    hxy = np.hypot(x[:,:,0], x[:,:,1])
    r = np.hypot(hxy, x[:,:,2])
    el = np.arctan2(x[:,:,2], hxy)
    az = np.arctan2(x[:,:,1], x[:,:,0])
    
    return r, el, az

def getFreqDomain(W,X,Y,Z, samplerate, win_type, win_length):
    W_fq = sig.stft(W, samplerate, win_type, win_length)
    X_fq = sig.stft(X, samplerate, win_type, win_length)
    Y_fq = sig.stft(Y, samplerate, win_type, win_length)
    Z_fq = sig.stft(Z, samplerate, win_type, win_length)
    t = W_fq[0]
    f = W_fq[1]
    
    return t, f, W_fq[2], X_fq[2], Y_fq[2], Z_fq[2]

def getXprime(X_fq, Y_fq, Z_fq):
    Xprime_Size = [np.shape(X_fq)[0],np.shape(X_fq)[1], 3]
    
    Xprime = np.empty(Xprime_Size, dtype=np.complex128)
    Xprime[:,:,0] = X_fq
    Xprime[:,:,1] = Y_fq
    Xprime[:,:,2] = Z_fq
    
    return Xprime, Xprime_Size

def getIntVec(W_fq, Xprime, Int_Size, Zo):
    I = np.empty(Int_Size, dtype=np.complex128)

    for i in range(0,3):
        I[:,:,i] = (W_fq*np.conj(Xprime[:,:,i]))
    
    I = np.abs(I)
    
    I[:,:,np.arange(3)] = -1/(2*np.sqrt(2)*Zo)*I[:,:,np.arange(3)]

    return I

def DOA(I, doa_Size):
    I_norm = np.linalg.norm(I, axis=2)
    
    doa = np.empty(doa_Size)
    for i in range(0,3):
        doa[:,:,i] = -(np.divide(I[:,:,i]+1e-10, I_norm+1e-10)) 
    
    r, el, az = cart2Sph(doa)
    
    return doa, r, el, az

def Diffuseness(Xprime, W_fq ):
    #norm_Xprime = np.empty(W_fq.shape, dtype=np.complex128)
    norm_Xprime = np.linalg.norm(Xprime, axis=2) 
    E = (np.power(norm_Xprime,2)/2 + np.power(np.absolute(W_fq),2))
    
    I_data = np.empty(Xprime_Size, dtype=np.complex128)
    for i in range(0,3):
        I_data[:,:,i] = (W_fq*np.conj(Xprime[:,:,i]))
    
    I_data = np.abs(I_data)
    
    diffueseness = np.empty(W_fq.shape)
    #diffueseness = 1 - (np.sqrt(2)* )
    for x in range(0,W_fq.shape[0]):
        for y in range(0,W_fq.shape[1]):
            if y < 9:
                avg_data = I_data[x,0:y+1,:]
                avg_data2 = E[x,0:y+1]
                diffueseness[x,y] = 1 - ((np.sqrt(2)* np.linalg.norm(np.average(avg_data)))/np.average(avg_data2))  
                
            else:
                avg_data = I_data[x,y-9:y+1,:]
                avg_data2 = E[x,y-9:y+1]
                diffueseness[x,y] = 1 - ((np.sqrt(2)* np.linalg.norm(np.average(avg_data)))/np.average(avg_data2))
                
    return diffueseness

def readGroundTruth():
    path = getBFormatAudioPath('groundTruth.txt')
    file = open(path, 'r')
    i = 0
    for line in file:
        if i == 2:
            azLine = line.split(": ")
            azimuth = azLine[1]
            azimuth = int(azimuth)
        
        elif i == 3:
            elLine = line.split(": ")
            elevation = elLine[1]
            elevation = int(elevation)
            
        i=i+1

    file.close()
    return azimuth, elevation
#%% Get Path and read audio file
    
bformat_pth = getBFormatAudioPath('bucket_FUMA_FUMA(0, 0).wav')

#Read audio file
data, samplerate = sf.read(bformat_pth)

#We get each channel individually
W = data[:,0]
X = data[:,1]
Y = data[:,2]
Z = data[:,3]

plotSignal('Waveform W',W)
plotSignal('Waveform X',X)
plotSignal('Waveform Y',Y)
plotSignal('Waveform Z',Z)

#%% We use STFT to get frequency domain of each channel

t, f, W_fq, X_fq, Y_fq, Z_fq = getFreqDomain(W,X,Y,Z,samplerate,'hann',256)


#plotSpectrogram
W_fq_db = to_dB(W_fq, 'N')
X_fq_db = to_dB(X_fq, 'N')
Y_fq_db = to_dB(Y_fq, 'N')
Z_fq_db = to_dB(Z_fq, 'N')

plotSpectrogram('Spectrogram W', W_fq_db,'viridis')
plotSpectrogram('Spectrogram X', X_fq_db,'viridis')
plotSpectrogram('Spectrogram Y', Y_fq_db,'viridis')
plotSpectrogram('Spectrogram Z', Z_fq_db,'viridis')

Xprime, Xprime_Size = getXprime(X_fq, Y_fq, Z_fq)

#%% Compute the intensity vector and DOA
ro = 1.204
c = 343
Zo = c*ro

I = getIntVec(W_fq, Xprime, Xprime_Size, Zo)

I_db = to_dB(I,'Y')

plotSpectrogram('Intensity X', I_db[:,:,0], 'viridis')
plotSpectrogram('Intensity Y', I_db[:,:,1], 'viridis')
plotSpectrogram('Intensity Z', I_db[:,:,2], 'viridis')

doa, r, el, az = DOA(I, Xprime_Size)

plotSpectrogram('Azimuth', az,'hsv')
plotSpectrogram('Elevation', el, 'viridis')
#%% Diffuseness computation

diffuseness = Diffuseness(Xprime, W_fq )

plotSpectrogram('Diffuseness', diffuseness, 'viridis')

            
azimuth, elevation = readGroundTruth()