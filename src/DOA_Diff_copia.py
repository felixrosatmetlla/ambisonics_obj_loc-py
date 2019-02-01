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
    plt.pcolormesh(np.abs(x_fq), cmap = colorMap)

    plt.colorbar()
    
def plotDOA(azimuth, elevation):
    #Plot Azimuth
    plt.figure()
    plt.suptitle('DOA: Azimuth')
    plt.pcolormesh(np.abs(azimuth), cmap = 'hsv', vmin = -np.pi, vmax = np.pi)

    plt.colorbar()
    
    #Plot Elevation
    plt.figure()
    plt.suptitle('DOA: Elevation')
    plt.pcolormesh(np.abs(elevation), cmap = 'viridis',  vmin = -np.pi/2, vmax = np.pi/2)

    plt.colorbar()
    
def to_dB(x, intensity):
    signal = np.abs(x.real) + 1e-100
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

def getEnergyVect(stft):
    p_kn = getPressure(stft)
    u_kn = getVelocityVect(stft)
    s1 = np.power(np.linalg.norm(u_kn,axis=0), 2)
    s2 = np.power(abs(p_kn), 2)

    data = ((p0/4.)*s1) + ((1./(4*p0*np.power(c,2)))*s2)
    return data

def getIntVec(stft):
    p_kn = getPressure(stft)
    u_kn = getVelocityVect(stft)
    
    #Xprime_Size = [np.shape(p_kn)[0],np.shape(p_kn)[1], 3]
    #I = np.empty(Xprime_Size, dtype=np.complex128)
    
    ch, f, t = np.shape(stft)
    I = np.zeros((ch-1,f,t))
    I = 0.5*np.real(p_kn * np.conj(u_kn))
    
    return I

def DOA(stft):
    i_kn = getIntVec(stft)
    
    I_norm = np.linalg.norm(i_kn, axis=0)
    doa = np.empty(np.shape(i_kn))
    for i in range(0,3):
        doa[i,:,:] = -(np.divide(i_kn[i,:,:], I_norm)) 
    
    r, el, az = cart2Sph(doa)
    
    return doa, r, el, az

def Diffuseness(stft, dt=10): 
    e_kn = getEnergyVect(stft)
    i_kn = getIntVec(stft)
    
    K = e_kn.shape[0]
    N = e_kn.shape[1]
    diffuseness = np.zeros((K, N))
    for k in range(K):
        for n in range(int(dt / 2), int(N - dt / 2)):
            num = np.linalg.norm(np.mean(i_kn[:, k, n:n + dt],axis = 1))
            den = c * np.mean(e_kn[k,n:n+dt])
            diffuseness[k,n] = 1-((num)/(den))
            #diffueseness[k,n] = 1 - ((num+1e-10)/(den+1e-10))
        
        # Borders: copy neighbor values
        for n in range(0, int(dt/2)):
            diffuseness[k, n] = diffuseness[k, int(dt/2)]
        
        for n in range(int(N - dt / 2), N):
            diffuseness[k, n] = diffuseness[k, N - int(dt/2) - 1]

    return diffuseness

def getMask(data, diffuseness, thr):
    mask = np.empty(np.shape(data))
    
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if (diffuseness[x,y]) < thr:
                mask[x,y] = 1
            else:
                mask[x,y] = np.nan
    return mask

def elMeanDev(data, mask):
    
    mask = getMask(data, diffuseness, 0.1)
    plotSpectrogram('eleMask', mask, 'viridis');
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

def azMeanDev(data, difuseness):
    
    mask = getMask(data, diffuseness, 0.1)
    plotSpectrogram('aziMask', mask, 'viridis');
    
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

def getMSE(data, diffuseness, gT):
    
    mask = getMask(data, diffuseness, 0.1)
    
    i=1
    aux = 0
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux = aux + np.power((gT - data[x,y]),2)
                i = i+1
    
    MSE = aux/i
    
    return MSE

def plotHist2D(azi, ele, diffuseness):
    
    mask = getMask(azi, diffuseness, 0.1)
    
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
    
bformat_pth = getBFormatAudioPath('drums_FUMA_FUMA(180, 0).wav')

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

#%% Compute the intensity vector and DOA

doa, r, el, az = DOA(stft)
plotDOA(az,el)

#%% Diffuseness computation

diffuseness = Diffuseness(stft, dt=10)
plotSpectrogram('Diffuseness', diffuseness, 'plasma_r')

#%%

azimuth_gt, elevation_gt = readGroundTruth()

azMean, azDev = azMeanDev(az,diffuseness)
elMean, elDev = elMeanDev(el,diffuseness)

azMSE = getMSE(az,diffuseness, azimuth_gt)
elMSE = getMSE(el,diffuseness, elevation_gt)

#%%

plotHist2D(az, el, diffuseness)
plotHist2DwMask(az, el)

#%%

writeResults('drums_FUMA_FUMA(180, 0).wav', azimuth_gt, elevation_gt, azMean, azDev, elMean, elDev, azMSE, elMSE)