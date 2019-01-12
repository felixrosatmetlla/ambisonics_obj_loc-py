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

#%% Global Variables

c = 346.13  # m/s
p0 = 1.1839 # kg/m3

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
    el = np.arcsin(x[:,:,2], hxy)
    az = np.arctan2(x[:,:,1], x[:,:,0])
    
    return r, el, az

def getFreqDomain(data, samplerate, win_type, win_length):
    
    for x in range(0,4):
        aux = sig.stft(data[:,x], samplerate, win_type, win_length)
        stft[:,:,x] = aux[2]

    t = aux[0]
    f = aux[1]
    
    return t, f, stft

def getVelocityVect(stft):    
    scale = -1.0/(p0*c)
    u_kn = scale*stft[:,:,1:]
    
    return u_kn

def getPressure(stft):
    p_kn = stft[:,:,0]
    
    return p_kn

def getEnergyVeect(p_kn, u_kn):
    norm_u_kn = np.linalg.norm(u_kn, axis=2) 
    e_kn = p0*np.power(norm_u_kn,2)/2. + ((1./(2*p0*np.power(c,2)))*np.power(np.absolute(p_kn),2))
    
    return e_kn

def getIntVec(p_kn, u_kn):
    
    for x in range(0,3):
        i_kn[:,:,x] = 0.5 * np.real(p_kn * np.conj(u_kn[:,:,x]))
    
    return i_kn

def DOA(I, doa_Size):
    I_norm = np.linalg.norm(I, axis=2)
    
    doa = np.empty(doa_Size)
    for i in range(0,3):
        doa[:,:,i] = -(np.divide(I[:,:,i]+1e-10, I_norm+1e-10)) 
    
    r, el, az = cart2Sph(doa)
    
    return doa, r, el, az

def Diffuseness(u_kn, W_fq ,I, dt=10):
    #norm_Xprime = np.empty(W_fq.shape, dtype=np.complex128)
    norm_u_kn = np.linalg.norm(u_kn, axis=2) 
    
    e_kn = p0*np.power(norm_u_kn,2)/2. + ((1./(2*p0*np.power(c,2)))*np.power(np.absolute(W_fq),2))
 
    i_kn = I
    
    """for i in range(0,3):
        I_data[:,:,i] = (W_fq*np.conj(Xprime[:,:,i]))
    
    I_data = np.abs(I_data)
    """
    diffueseness = np.empty(W_fq.shape)
    #diffueseness = 1 - (np.sqrt(2)* )
    """for x in range(0,W_fq.shape[0]):
        for y in range(0,W_fq.shape[1]):
            if y < 9:
                avg_data = I_data[x,0:y+1,:]
                avg_data2 = E[x,0:y+1]
                diffueseness[x,y] = 1 - ((np.sqrt(2)* np.linalg.norm(np.mean(avg_data)))/(c*np.mean(avg_data2)))  
                
            else:
                avg_data = I_data[x,y-9:y+1,:]
                avg_data2 = E[x,y-9:y+1]
                diffueseness[x,y] = 1 - ((np.sqrt(2)* np.linalg.norm(np.mean(avg_data)))/(c*np.mean(avg_data2)))
                
    """
    K = W_fq.shape[0]
    N = W_fq.shape[1]
    dif = np.zeros((K, N))

    for k in range(K):
        for n in range(int(dt / 2), int(N - dt / 2)):
            num = np.linalg.norm(np.mean(i_kn[:, k, n:n + dt], axis=1))
            den = c * np.mean(e_kn[k,n:n+dt])
            dif[k,n] = 1 - (num/(den+1e-10))

        # Borders: copy neighbor values
        for n in range(0, int(dt/2)):
            dif[k, n] = dif[k, int(dt/2)]

        for n in range(int(N - dt / 2), N):
            dif[k, n] = dif[k, N - int(dt/2) - 1]
            
    return diffueseness

def readGroundTruth():
    path = getBFormatAudioPath('groundTruth.txt')
    file = open(path, 'r')
    i = 0
    for line in file:
        if i == 3:
            azLine = line.split(": ")
            azimuth = azLine[1]
            azimuth = int(azimuth)
        
        elif i == 4:
            elLine = line.split(": ")
            elevation = elLine[1]
            elevation = int(elevation)
            
        i=i+1

    file.close()
    return azimuth, elevation
#%% Get Path and read audio file
    
bformat_pth = getBFormatAudioPath('violin_FUMA_FUMA(0, 0).wav')

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

time, freq, stft = getFreqDomain(data,samplerate,'hann',256)

#plotSpectrogram
W_fq_db = to_dB(stft[:,:,0], 'N')
X_fq_db = to_dB(stft[:,:,1], 'N')
Y_fq_db = to_dB(stft[:,:,2], 'N')
Z_fq_db = to_dB(stft[:,:,3], 'N')

plotSpectrogram('Spectrogram W', W_fq_db,'viridis')
plotSpectrogram('Spectrogram X', X_fq_db,'viridis')
plotSpectrogram('Spectrogram Y', Y_fq_db,'viridis')
plotSpectrogram('Spectrogram Z', Z_fq_db,'viridis')

u_kn = getVelocityVect(stft)

#%% Compute the intensity vector and DOA
I = getIntVec(stft[:,:,0], u_kn)

I_db = to_dB(I,'Y')

plotSpectrogram('Intensity X', I_db[:,:,0], 'viridis')
plotSpectrogram('Intensity Y', I_db[:,:,1], 'viridis')
plotSpectrogram('Intensity Z', I_db[:,:,2], 'viridis')

doa, r, el, az = DOA(I, Xprime_Size)

plotSpectrogram('Azimuth', az,'hsv')
plotSpectrogram('Elevation', el, 'viridis')
#%% Diffuseness computation

diffuseness = Diffuseness(u_kn, stft[:,:,0] ,I,dt=10)

plotSpectrogram('Diffuseness', diffuseness, 'viridis')

            
azimuth, elevation = readGroundTruth()