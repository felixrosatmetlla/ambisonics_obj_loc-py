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
import xml.etree.ElementTree as ET
import random

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
    
def getOutputAudioPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test/output/'+ output_filename) 
    return output_path

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
    hxy = np.hypot(x[0,:,:], x[1,:,:])
    r = np.hypot(hxy, x[2,:,:])
    el = np.arcsin(x[2,:,:], hxy)
    az = np.arctan2(x[1,:,:], x[0,:,:])
    
    return r, el, az

def getFreqDomain(data, samplerate, win_type, win_length):
    aux = sig.stft(data[:,0], samplerate, win_type, win_length)
    Xprime_Size = [4,np.shape(aux[2])[0],np.shape(aux[2])[1]]
    stft = np.empty(Xprime_Size, dtype=np.complex128)
    for x in range(0,4):
        aux = sig.stft(data[:,x], samplerate, win_type, win_length)
        stft[x,:,:] = aux[2]

    t = aux[0]
    f = aux[1]
    
    return t, f, stft

def getVelocityVect(stft):    
    scale = -1.0/(np.sqrt(2)*p0*c)
    u_kn = scale*stft[1:,:,:]
    
    return u_kn

def getPressure(stft):
    p_kn = stft[0,:,:]
    
    return p_kn

def getEnergyVect(p_kn, u_kn):

    s1 = np.power(np.linalg.norm(u_kn,axis=0), 2)
    s2 = np.power(abs(p_kn), 2)

    data = ((p0/4.)*s1) + ((1./(4*p0*np.power(c,2)))*s2)
    return data

def getIntVec(p_kn, u_kn):
    
    Xprime_Size = [np.shape(p_kn)[0],np.shape(p_kn)[1], 3]
    I = np.empty(Xprime_Size, dtype=np.complex128)

    I = np.real(p_kn * np.conj(u_kn))
    
    return I

def DOA(I):
    I_norm = np.linalg.norm(I, axis=0)
    doa = np.empty(np.shape(I))
    for i in range(0,3):
        doa[i,:,:] = -(np.divide(I[i,:,:], I_norm)) 
    
    r, el, az = cart2Sph(doa)
    
    return doa, r, el, az

def Diffuseness(u_kn, W_fq ,I, dt=10):
    
    e_kn=getEnergyVect(W_fq, u_kn)
    i_kn = I
    
    """for i in range(0,3):
        I_data[:,:,i] = (W_fq*np.conj(Xprime[:,:,i]))
    
    I_data = np.abs(I_data)
    """
    #diffueseness = np.empty(W_fq.shape)
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
                
    
    K = W_fq.shape[0]
    N = W_fq.shape[1]
    diffueseness = np.zeros((K, N))
     
    for k in range(K):
        for n in range(5, (N-5)):
            num = np.linalg.norm(np.mean(i_kn[ k, n:n + dt,:]))
            den = c * np.mean(e_kn[k,n:n+dt])
            diffueseness[k,n] = 1 - (num/(den+1e-10))

        # Borders: copy neighbor values
        for n in range(0, 5):
            diffueseness[k, n] = diffueseness[k, 5]

        for n in range((N - 5), N):
            diffueseness[k, n] = diffueseness[k, N - 5 - 1]
    """
    K = W_fq.shape[0]
    N = W_fq.shape[1]
    diffueseness = np.zeros((K, N))
    for k in range(K):
        for n in range(int(dt / 2), int(N - dt / 2)):
            num = np.linalg.norm(np.mean(i_kn[:, k, n:n + dt]))
            den = c * np.mean(e_kn[k,n:n+dt])
            diffueseness[k,n] = 1-((num)/(den))
            #diffueseness[k,n] = 1 - ((num+1e-10)/(den+1e-10))
        
        # Borders: copy neighbor values
        for n in range(0, int(dt/2)):
            diffueseness[k, n] = diffueseness[k, int(dt/2)]
        
        for n in range(int(N - dt / 2), N):
            diffueseness[k, n] = diffueseness[k, N - int(dt/2) - 1]

    return diffueseness

def getMask(data, diffuseness, thr):
    mask = np.empty(np.shape(data))
    
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if (1-diffuseness[x,y]) > thr:
                mask[x,y] = 1
            else:
                mask[x,y] = np.nan
    return mask

def elMeanDev(data, mask):
    i=1
    aux = 0
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux = aux + data[x,y]
                i = i+1
                
    mean = aux/i
    
    j=1
    aux2 = 0
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux2 = aux2 + np.power((data[x,y]-mean),2)
                j = j+1
    
    dev = np.sqrt(aux2/(j-1))
    return mean, dev

def azMeanDev(data, mask):
    
    i=1
    aux_c = 0
    aux_s = 0 
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux_c = aux_c + np.cos(data[x,y])
                aux_s = aux_s + np.sin(data[x,y])
                i = i+1
    C = aux_c/i
    S = aux_s/i
    mean = np.arctan2(S,C)
    
    R = np.sqrt((np.power(C,2) + np.power(S,2)))
    dev = np.sqrt(-2*np.log(R))
    return mean, dev

def getMSE(data, mask, gT):
    
    i=1
    aux = 0
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux = aux + np.power((gT - data[x,y]),2)
                i = i+1
    
    MSE = aux/i
    
    return MSE

def plotHist2D(azi, ele, mask):
    plt.figure()
    #plt.suptitle(title)
    i=0
    azimuth = np.empty(1)
    elevation = np.empty(1)
    for x in range(np.shape(azi)[0]):
        for y in range(np.shape(azi)[1]):
            if mask[x,y] == 1:
                if i == 0:
                    azimuth[i] = azi[x,y]
                    elevation[i] = ele[x,y]
                    i = i+1
                else:
                    azimuth = np.append(azimuth ,azi[x,y])
                    elevation = np.append(elevation, ele[x,y])
                    i = i+1
                
    plt.hist2d(azimuth, elevation, bins= [360, 180])
    plt.colorbar()
    
    plt.figure()
    nbins = [360, 180]
    H, xedges, yedges = np.histogram2d(azimuth,elevation,bins=nbins)
     
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    
def plotHist2DwMask(azi, ele):
    plt.figure()
    #plt.suptitle(title)
    i=0
    azimuth = np.empty(1)
    elevation = np.empty(1)
    for x in range(np.shape(azi)[0]):
        for y in range(np.shape(azi)[1]):
            if i == 0:
                azimuth[i] = azi[x,y]
                elevation[i] = ele[x,y]
                i = i+1
            else:
                azimuth = np.append(azimuth ,azi[x,y])
                elevation = np.append(elevation, ele[x,y])
                i = i+1
                
    plt.hist2d(azimuth, elevation, bins= [360, 180])
    plt.colorbar()
    
    plt.figure()
    nbins = [360, 180]
    H, xedges, yedges = np.histogram2d(azimuth,elevation,bins=nbins)
     
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    
def readGroundTruth():
    path = getBFormatAudioPath('groundTruth.xml')
    tree = ET.parse(path)
    root = tree.getroot()
    
    azimuth = root.findtext('azimuth')
    elevation = root.findtext('elevation')
    
    return float(azimuth), float(elevation)

def writeResults(filename, azimuth, elevation, azMean, azDev, elMean, elDev, azMSE, elMSE):
    data = ET.Element('data')
    title = ET.SubElement(data,'title')
    filenm = ET.SubElement(data, 'filename')
    azimuth_xml = ET.SubElement(data, 'azimuth')
    elevation_xml = ET.SubElement(data, 'elevation')
    
    results = ET.SubElement(data,'results')
    azimuth_ = ET.SubElement(results, 'azimuth')
    aziMean = ET.SubElement(azimuth_, 'Mean')
    aziDev = ET.SubElement(azimuth_, 'Deviation')
    aziMSE = ET.SubElement(azimuth_, 'MSE')
    elevation_ = ET.SubElement(results, 'elevation')
    eleMean = ET.SubElement(elevation_, 'Mean')
    eleDev = ET.SubElement(elevation_, 'Deviation')
    eleMSE = ET.SubElement(elevation_, 'MSE')
    
    title.set('name','GroundTruth')
    filenm.set('name','Filename')
    azimuth_xml.set('name','Azimuth')
    elevation_xml.set('name','Elevation')
    results.set('name','Results')
    azimuth_.set('name','Azimuth')
    elevation_.set('name','Elevation')
    aziMean.set('name','Azimuth Mean')
    aziDev.set('name','Azimuth Deviation')
    aziMSE.set('name','Azimuth MSE')
    eleMean.set('name','Elevation Mean')
    eleDev.set('name','Elevation Deviation')
    eleMSE.set('name','Elevation MSE')
    
    title.text = 'Ground Truth'
    filenm.text = filename
    azimuth_xml.text = str(azimuth)
    elevation_xml.text = str(elevation)
    aziMean.text = str(azMean)
    aziDev.text = str(azDev)
    aziMSE.text = str(azMSE)
    eleMean.text = str(elMean)
    eleDev.text = str(elDev)
    eleMSE.text = str(elMSE)
    
    path = getOutputAudioPath("results.xml")
    results = ET.ElementTree(data)
    results.write(path)
    
#%% Get Path and read audio file
    
bformat_pth = getBFormatAudioPath('violin_FUMA_FUMA(0, 0).wav')

#Read audio file
data, samplerate = sf.read(bformat_pth)

for i in range(np.shape(data)[0]):
    for n in range(4):
        data[i,n] += (random.random()-0.5)*1e-2
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
W_fq_db = to_dB(stft[0,:,:], 'N')
X_fq_db = to_dB(stft[1,:,:], 'N')
Y_fq_db = to_dB(stft[2,:,:], 'N')
Z_fq_db = to_dB(stft[3,:,:], 'N')

plotSpectrogram('Spectrogram W', W_fq_db,'viridis')
plotSpectrogram('Spectrogram X', X_fq_db,'viridis')
plotSpectrogram('Spectrogram Y', Y_fq_db,'viridis')
plotSpectrogram('Spectrogram Z', Z_fq_db,'viridis')

u_kn = getVelocityVect(stft)
p_kn = getPressure(stft)
#%% Compute the intensity vector and DOA
I = getIntVec(p_kn, u_kn)

I_db = to_dB(I,'Y')

plotSpectrogram('Intensity X', I_db[0,:,:], 'viridis')
plotSpectrogram('Intensity Y', I_db[1,:,:], 'viridis')
plotSpectrogram('Intensity Z', I_db[2,:,:], 'viridis')

doa, r, el, az = DOA(I)

plotSpectrogram('Azimuth', az,'hsv')
plotSpectrogram('Elevation', el, 'viridis')
#%% Diffuseness computation

diffuseness = Diffuseness(u_kn, stft[0,:,:] ,I,dt=10)

plotSpectrogram('Diffuseness', diffuseness, 'viridis')

#%%
            
azimuth, elevation = readGroundTruth()

elMask = getMask(el, diffuseness, 0.6)
azMask = getMask(az, diffuseness, 0.6)

plotSpectrogram('Elevation Mask', elMask, 'viridis')
plotSpectrogram('Azimuth Mask', azMask, 'viridis')

elMean, elDev = elMeanDev(el,elMask)
azMean, azDev = azMeanDev(az,azMask)

#%%

plotHist2D(az, el, elMask)
plotHist2DwMask(az, el)

#%%

azMSE = getMSE(az,azMask, azimuth)
elMSE = getMSE(el,elMask, elevation)

writeResults('violin_FUMA_FUMA(0, 0).wav', azimuth, elevation, azMean, azDev, elMean, elDev, azMSE, elMSE)