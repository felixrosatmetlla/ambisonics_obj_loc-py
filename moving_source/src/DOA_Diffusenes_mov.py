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
import scipy as sc
from mpl_toolkits import mplot3d
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
    output_path = os.path.join(output_path,'test/output/BFormat_Audios/'+ output_filename) 
    return output_path

def getGTPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test/output/Ground_Truth/'+ output_filename) 
    return output_path

def getResultsPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test/output/Results/'+ output_filename) 
    return output_path

def plotSignal(title, x):
    plt.figure()
    plt.suptitle(title)
    plt.plot(x)

def plotSpectrogram(title, x_fq, colorMap):

    plt.figure()
    plt.suptitle(title)
    plt.pcolormesh(np.abs(x_fq), cmap = colorMap)

    plt.colorbar()
    
def plotDOA(azimuth, elevation):
    #Plot Azimuth
    plt.figure()
    plt.suptitle('DOA: Azimuth')
    plt.pcolormesh((azimuth), cmap = 'hsv', vmin = -np.pi, vmax = np.pi)

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
            if (diffuseness[x,y]) <= thr:
                mask[x,y] = 1
            else:
                mask[x,y] = np.nan
    return mask

def elMeanDev(data, diffuseness, threshold):
    
    mask = getMask(data, diffuseness, threshold)
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

def azMeanDev(data, diffuseness, threshold):
    
    mask = getMask(data, diffuseness, threshold)
    plotSpectrogram('aziMask', mask, 'viridis');
    
    i=1
    aux_mean = np.array([[]])
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux_mean = np.concatenate((aux_mean, np.array([[mask[x,y]]])),axis=1)
                i = i+1
    

    mean = sc.stats.circmean(aux_mean,high=2*np.pi,low=0)
    

    dev = sc.stats.circstd(aux_mean,high=2*np.pi,low=0)
    return mean, dev

def getMSE(data, diffuseness, gT, threshold):
    
    mask = getMask(data, diffuseness, threshold)
    
    i=1
    aux = 0
    for x in range(np.shape(data)[0]):
        for y in range(np.shape(data)[1]):
            if mask[x,y] == 1:
                aux = aux + np.power((gT - data[x,y]),2)
                i = i+1
    
    MSE = aux/i
    
    return MSE

def plotHist2D(azi, ele, diffuseness, threshold):
    
    mask = getMask(azi, diffuseness, threshold)
    
    plt.figure()
    plt.suptitle('DOA Histogram 2D w/ Mask')
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
    plt.suptitle('DOA Histogram 2D w/Mask')
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
    plt.suptitle('DOA Histogram 2D without mask')
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
    plt.suptitle('DOA Histogram 2D without mask')
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
    
def readGroundTruth(filename):
    file = filename.split(".")
    path = getGTPath('groundTruth_'+ file[0] + '.xml')
    tree = ET.parse(path)
    root = tree.getroot()
    
    azimuth = root.findtext('azimuth')
    elevation = root.findtext('elevation')
    
    return float(azimuth), float(elevation)

def writeResults(filename, azimuth, elevation, azMean, azDev, elMean, elDev, azMSE, elMSE, thresh):
    data = ET.Element('data')
    title = ET.SubElement(data,'title')
    filenm = ET.SubElement(data, 'filename')
    azimuth_xml = ET.SubElement(data, 'azimuth')
    elevation_xml = ET.SubElement(data, 'elevation')
    threshold = ET.SubElement(data, 'threshold')
    
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
    threshold.set('name','Threshold')
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
    threshold.text = str(thresh)
    aziMean.text = str(azMean)
    aziDev.text = str(azDev)
    aziMSE.text = str(azMSE)
    eleMean.text = str(elMean)
    eleDev.text = str(elDev)
    eleMSE.text = str(elMSE)
    
    file = filename.split(".")
    path = getResultsPath("results_" + file[0] + ".xml")
    results = ET.ElementTree(data)
    results.write(path)
    
def addNoise(data, noise):
    for i in range(np.shape(data)[0]):
        for n in range(4):
            data[i,n] += (random.random()-0.5)*noise
    
    return data

def getDoaResults(filename, thr, noise):
    bformat_pth = getBFormatAudioPath(filename)
    
    #Read audio file
    data, samplerate = sf.read(bformat_pth)
    data = addNoise(data,1e-5)
    time, freq, stft = getFreqDomain(data,samplerate,'hann',256)
    
    doa, r, el, az = DOA(stft)
    diffuseness = Diffuseness(stft, dt=10)
    
    azimuth_gt, elevation_gt = readGroundTruth(filename)

    azMean, azDev = azMeanDev(az,diffuseness, thr)
    elMean, elDev = elMeanDev(el,diffuseness, thr)
    
    azMSE = getMSE(az,diffuseness, azimuth_gt, thr)
    elMSE = getMSE(el,diffuseness, elevation_gt, thr)   
     
    writeResults('drums_FUMA_FUMA(180, 0)_%d.wav'%(thr), azimuth_gt, elevation_gt, azMean, azDev, elMean, elDev, azMSE, elMSE, thr)
    
    MSE = [azMSE, elMSE]
    return MSE
    
def PlotMSEVariables(mse_results, threshold, noise):
    plt.figure()
    plt.grid()
    plt.suptitle('Azimuth MSE respect Threshold')
    for nse in range (len(noise)):
        thr_az = mse_results[:,nse,0]
        plt.plot(threshold,thr_az,label="Noise %f"%(noise[nse]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.figure()
    plt.grid()
    plt.suptitle('Elevation MSE respect Threshold')
    for nse in range (len(noise)):
        thr_el = mse_results[:,nse,1]
        plt.plot(threshold,thr_el,label="Noise %f"%(noise[nse]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.figure()
    plt.grid()
    plt.suptitle('Azimuth MSE respect Noise')
    for thr in range (len(threshold)):
        noise_az = mse_results[thr,:,0]
        plt.plot(noise,noise_az,label="Threshold %f"%(threshold[thr]))
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.figure()
    plt.grid()
    plt.suptitle('Elevation MSE respect Noise')
    for thr in range (len(threshold)):
        noise_el = mse_results[thr,:,1]
        plt.plot(noise,noise_el,label="Threshold %f"%(threshold[thr]))
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    from mpl_toolkits.mplot3d import axes3d    
    
    th, ns = np.meshgrid(threshold, noise)
    
    # Plot a basic wireframe.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.invert_yaxis()
    wireframe = ax.plot_wireframe(np.transpose(th),np.transpose(ns), mse_results[:,:,0])
    plt.show()

        
    
    
    
#%% Get Path and read audio file

filename = 'drums_FUMA_FUMA(0, 45).wav';
bformat_pth = getBFormatAudioPath(filename)

#Read audio file
data, samplerate = sf.read(bformat_pth)

data = addNoise(data,1e-5)

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

freq, time, stft = getFreqDomain(data,samplerate,'hann',256)

#plotSpectrogram
W_fq_db = to_dB(stft[0,:,:], 'N')
X_fq_db = to_dB(stft[1,:,:], 'N')
Y_fq_db = to_dB(stft[2,:,:], 'N')
Z_fq_db = to_dB(stft[3,:,:], 'N')

plotSpectrogram('Spectrogram W', W_fq_db,'viridis')
plotSpectrogram('Spectrogram X', X_fq_db,'viridis')
plotSpectrogram('Spectrogram Y', Y_fq_db,'viridis')
plotSpectrogram('Spectrogram Z', Z_fq_db,'viridis')
#%%
doas, rs, els, azs = DOA(stft)
plotDOA(azs,els)

N = stft.shape[2]
mov_azi = np.zeros(N)


mov_azi = np.mean(azs, axis=0)
plt.figure()     
plt.plot(mov_azi)    
#%% Compute the intensity vector and DOA

K = stft.shape[1]
N = stft.shape[2]



#for t in range(N):
#    if(t<10):
#        doas, rs, els, azs = DOA(stft[:,:,0:(t+10)])
#    elif(t>N-10):
#        doas, rs, els, azs = DOA(stft[:,:,(t-10):(N-1)])
#    else:
#        doas, rs, els, azs = DOA(stft[:,:,(t-10):(t+10)])
#    plotDOA(azs,els)

#%% Diffuseness computation

diffuseness = Diffuseness(stft, dt=10)
plotSpectrogram('Diffuseness', diffuseness, 'plasma_r')

#%%

azimuth_gt, elevation_gt = readGroundTruth(filename)

thr = 0.1

azMean, azDev = azMeanDev(az,diffuseness, thr)
elMean, elDev = elMeanDev(el,diffuseness, thr)

azMSE = getMSE(az,diffuseness, azimuth_gt, thr)
elMSE = getMSE(el,diffuseness, elevation_gt, thr)

#%%

plotHist2D(az, el, diffuseness, thr)
plotHist2DwMask(az, el)

#%%

writeResults('drums_FUMA_FUMA(180, 0)_%d.wav'%(thr), azimuth_gt, elevation_gt, azMean, azDev, elMean, elDev, azMSE, elMSE, thr)

#%%
filename = 'drums_FUMA_FUMA(0, 45).wav';

thresholds = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
index=0
mse_results = np.zeros((np.size(thresholds), np.size(noise),2))
for thr in range (len(thresholds)):
    for nse in range (len(noise)):
        print(index)
        mse_results[thr,nse, :] = getDoaResults(filename,thresholds[thr],noise[nse])
        index = index +1
#%%%
PlotMSEVariables(mse_results, thresholds, noise)      