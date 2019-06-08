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
import math
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

def getPlotPath(output_filename, thr, noise, azi_gt,ele_gt):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.dirname(output_path)
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'Memory/Plots/%.2f_%.2f/%.2f_%.6f'%(azi_gt,ele_gt,thr,noise))
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_path = os.path.join(output_path,output_filename)

    return output_path

def getResultsPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test/output/Results/'+ output_filename) 
    return output_path

def plotSignal(title, x, xlabel, ylabel, path):
    plt.figure()
    plt.suptitle(title)
    plt.plot(x)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.savefig(path, bbox_inches='tight')


def plotSpectrogram(title, x_fq, colorMap, xlabel, ylabel, barLabel,path):

    plt.figure()
    plt.suptitle(title)
    plt.pcolormesh(np.abs(x_fq), cmap = colorMap)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barLabel)
    
    plt.savefig(path, bbox_inches='tight')

    
def plotDOA(AziTitle,EleTitle, azimuth, elevation, xlabel, ylabel, barLabel_azi, barLabel_ele, AziPath, ElePath):
    #Plot Azimuth
    plt.figure()
    plt.suptitle(AziTitle)
    plt.pcolormesh(np.abs(azimuth), cmap = 'hsv', vmin = -np.pi, vmax = np.pi)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barLabel_azi)
    
    plt.savefig(AziPath, bbox_inches='tight')
    
    #Plot Elevation
    plt.figure()
    plt.suptitle(EleTitle)
    plt.pcolormesh(np.abs(elevation), cmap = 'viridis',  vmin = -np.pi/2, vmax = np.pi/2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barLabel_ele)
    
    plt.savefig(ElePath, bbox_inches='tight')
    
    
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
    #plotSpectrogram('eleMask', mask, 'viridis','testx', 'testy', 'testBar',getPlotPath('elmeandev.png', thr, noise));
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
    #plotSpectrogram('aziMask', mask, 'viridis','testx', 'testy', 'testBar',getPlotPath('azmeandev.png', thr, noise));
    
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

def plotHist2D(azi, ele, diffuseness, threshold, title, xlabel, ylabel, barLabel, path):
    
    mask = getMask(azi, diffuseness, threshold)
    
    plt.figure()
    plt.suptitle(title)
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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barLabel)
    
    plt.savefig(path+'_.png', bbox_inches='tight')
    
    plt.figure()
    plt.suptitle(title)
    nbins = [360, 180]
    H, xedges, yedges = np.histogram2d(azimuth,elevation,bins=nbins)
     
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barLabel)
    
    plt.savefig(path, bbox_inches='tight')

    
def plotHist2DwMask(azi, ele, title, xlabel, ylabel, barLabel, path):
    plt.figure()
    plt.suptitle(title)
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
    # Plot 2D histogram using pcolor
    
    plt.figure()
    plt.suptitle(title)
    nbins = [360, 180]
    H, xedges, yedges = np.histogram2d(azimuth,elevation,bins=nbins)
     
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barLabel)
    
    plt.savefig(path, bbox_inches='tight')
    
    
def readGroundTruth(filename):
    file = filename.split(".")
    path = getGTPath('groundTruth_'+ file[0] + '.xml')
    tree = ET.parse(path)
    root = tree.getroot()
    
    azimuth = root.findtext('azimuth')
    elevation = root.findtext('elevation')
    reverb = root.findtext('reverb')
    
    return float(azimuth), float(elevation), bool(reverb)

def writeResults(filename, azimuth, elevation,reverb, azMean, azDev, elMean, elDev, azMSE, elMSE, thresh, noise):
    data = ET.Element('data')
    title = ET.SubElement(data,'title')
    filenm = ET.SubElement(data, 'filename')
    reverb_xml = ET.SubElement(data, 'reverb')
    azimuth_xml = ET.SubElement(data, 'azimuth')
    elevation_xml = ET.SubElement(data, 'elevation')
    threshold = ET.SubElement(data, 'threshold')
    noise_xml = ET.SubElement(data, 'noise')
    
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
    reverb_xml.set('name', 'Reverb')
    azimuth_xml.set('name','Azimuth')
    elevation_xml.set('name','Elevation')
    threshold.set('name','Threshold')
    noise_xml.set('name', 'Noise')
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
    noise_xml.text = str(noise) 
    aziMean.text = str(azMean)
    aziDev.text = str(azDev)
    aziMSE.text = str(azMSE)
    eleMean.text = str(elMean)
    eleDev.text = str(elDev)
    eleMSE.text = str(elMSE)
    if(reverb == True):
        reverb_xml.text = str(reverb)
    elif(reverb == False):
        reverb_xml.text = str(reverb)
    
    file = filename.split(".")
    path = getResultsPath("results_" + file[0] + ".xml")
    results = ET.ElementTree(data)
    results.write(path)
    
def addNoise(data, noise):
    for i in range(np.shape(data)[0]):
        for n in range(4):
            data[i,n] += (random.random()-0.5)*noise
    
    return data

def getDoaResults(filename, thr, noise, azimuth_gt, elevation_gt):
    bformat_pth = getBFormatAudioPath(filename)
    
    #Read audio file
    data, samplerate = sf.read(bformat_pth)
    data = addNoise(data,1e-5)
    time, freq, stft = getFreqDomain(data,samplerate,'hann',256)
    
    doa, r, el, az = DOA(stft)
    plotDOA('Azimuth', 'Elevation',az,el,'Time', 'Frequency', 'Azimuth (rad)','Elevation (rad)',
            getPlotPath('azi_%f_%f.png'%(thr,noise), thr, noise,azimuth_gt, elevation_gt),getPlotPath('ele_%f_%f.png'%(thr,noise), thr, noise,azimuth_gt, elevation_gt) )
    
    diffuseness = Diffuseness(stft, dt=10)
    plotSpectrogram('Diffuseness', diffuseness, 'plasma_r','Time', 'Frequency', 'Diffuseness', getPlotPath('diff_%f_%f.png'%(thr,noise), thr, noise,azimuth_gt, elevation_gt))

#    r, elevation_gt, azimuth_gt = cart2sph(0,5,0.06)
    #azimuth_gt, elevation_gt, reverb = readGroundTruth(filename)

    azMean, azDev = azMeanDev(az,diffuseness, thr)
    elMean, elDev = elMeanDev(el,diffuseness, thr)
    
    plotHist2D(az, el, diffuseness, thr,'DoA Histogram with Mask', 'Azimuth', 'Elevation', 'Number Samples',getPlotPath('DoAHist_%f_%f.png'%(thr,noise), thr, noise, azimuth_gt, elevation_gt))
    plotHist2DwMask(az, el, 'DoA Histogram without Mask','Azimuth', 'Elevation', 'Number Samples',getPlotPath('DoAHist_wMask_%f_%f.png'%(thr,noise), thr, noise, azimuth_gt, elevation_gt))
    
    azMSE = getMSE(az,diffuseness, azimuth_gt, thr)
    elMSE = getMSE(el,diffuseness, elevation_gt, thr)   
    
    file = filename.split(".")
    resultsFilename = file[0] + '_thr'+ str(thr) + '_nse' + str(noise)+'.wav'
    writeResults(resultsFilename, azimuth_gt, elevation_gt, reverb, azMean, azDev, elMean, elDev, azMSE, elMSE, thr, noise)
    
    MSE = [azMSE, elMSE]
    return MSE
    
def PlotMSEVariables(mse_results, threshold, noise,xlabel_thr, xlabel_nse,zlabel, AzMSEThrTitle,ElMSEThrTitle,AzMSENseTitle,ElMSENseTitle,wireframeTitle,azi_gt, ele_gt):
    plt.figure()
    plt.grid()
    plt.suptitle(AzMSEThrTitle)
    for nse in range (len(noise)):
        thr_az = mse_results[:,nse,0]
        plt.plot(threshold,thr_az,label="Noise %f"%(noise[nse]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel(xlabel_thr)
    plt.ylabel(zlabel)
    
    plt.savefig(getPlotPath('Azi_MSE_noise.png', -256, -256,azi_gt, ele_gt), bbox_inches='tight')
 
    plt.figure()
    plt.grid()
    plt.suptitle(ElMSEThrTitle)
    for nse in range (len(noise)):
        thr_el = mse_results[:,nse,1]
        plt.plot(threshold,thr_el,label="Noise %f"%(noise[nse]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.xlabel(xlabel_thr)
    plt.ylabel(zlabel)
    plt.savefig(getPlotPath('Ele_MSE_noise.png', -256, -256,azi_gt, ele_gt), bbox_inches='tight')

    plt.figure()
    plt.grid()
    plt.suptitle(AzMSENseTitle)
    for thr in range (len(threshold)):
        noise_az = mse_results[thr,:,0]
        plt.plot(noise,noise_az,label="Threshold %f"%(threshold[thr]))
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel(xlabel_nse)
    plt.ylabel(zlabel)
    plt.savefig(getPlotPath('Azi_MSE_Threshold.png', -256, -256,azi_gt, ele_gt), bbox_inches='tight')

    plt.figure()
    plt.grid()
    plt.suptitle(ElMSENseTitle)
    for thr in range (len(threshold)):
        noise_el = mse_results[thr,:,1]
        plt.plot(noise,noise_el,label="Threshold %f"%(threshold[thr]))
        plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel(xlabel_nse)
    plt.ylabel(zlabel)
    plt.savefig(getPlotPath('Ele_MSE_Threshold.png', -256, -256,azi_gt, ele_gt), bbox_inches='tight')

    from mpl_toolkits.mplot3d import axes3d    
    
    th, ns = np.meshgrid(threshold, noise)
    
    # Plot a basic wireframe.
    fig = plt.figure()
    plt.suptitle(wireframeTitle)
    ax = fig.gca(projection='3d')
    ax.invert_yaxis()
    wireframe = ax.plot_wireframe(np.transpose(th),np.transpose(ns), mse_results[:,:,0])
    ax.set_xlabel(xlabel_thr)
    ax.set_ylabel(xlabel_nse)
    ax.set_zlabel(zlabel)
    
    plt.savefig(getPlotPath('MSE_Threshold_noise.png', -256, -256,azi_gt, ele_gt), bbox_inches='tight')

    plt.show()

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    
    return r, elev, az        
    
    
    
#%% Get Path and read audio file

filename = 'drums_FUMA_FUMA_r_False(-180, 0).wav';
bformat_pth = getBFormatAudioPath(filename)

#Read audio file
data, samplerate = sf.read(bformat_pth)

thr = 0.1
noise = 1e-5

data = addNoise(data,noise)

#%%
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

plotSpectrogram('Spectrogram W', W_fq_db,'viridis', 'testx', 'testy', 'testBar',getPlotPath('test.png', thr, noise))
plotSpectrogram('Spectrogram X', X_fq_db,'viridis', 'testx', 'testy', 'testBar',getPlotPath('test2.png', thr, noise))
plotSpectrogram('Spectrogram Y', Y_fq_db,'viridis', 'testx', 'testy', 'testBar',getPlotPath('test3.png', thr, noise))
plotSpectrogram('Spectrogram Z', Z_fq_db,'viridis', 'testx', 'testy', 'testBar',getPlotPath('test4.png', thr, noise))

#%% Compute the intensity vector and DOA

doa, r, el, az = DOA(stft)
#plotDOA('Azi', 'Ele',az,el, 'testx', 'testy', 'testBar',getPlotPath('azi.png', thr, noise),getPlotPath('ele.png', thr, noise))

#%% Diffuseness computation

diffuseness = Diffuseness(stft, dt=10)
#plotSpectrogram('Diffuseness', diffuseness, 'plasma_r', 'testx', 'testy', 'testBar',getPlotPath('diff.png', thr, noise))

#%%

#r, elevation_gt, azimuth_gt = cart2sph(-4.7,-1.71,0.06)
azimuth_gt, elevation_gt, reverb = readGroundTruth(filename)

#%%

azMean, azDev = azMeanDev(az,diffuseness, thr)
elMean, elDev = elMeanDev(el,diffuseness, thr)

azMSE = getMSE(az,diffuseness, azimuth_gt, thr)
elMSE = getMSE(el,diffuseness, elevation_gt, thr)

#%%

#plotHist2D(az, el, diffuseness, thr,'title', 'testx', 'testy', 'testBar',getPlotPath('hist.png', thr, noise))
#plotHist2DwMask(az, el, 'title','testx', 'testy', 'testBar',getPlotPath('hist2.png', thr, noise))

#%%
file = filename.split(".")
resultsFilename = file[0] + '_thr'+ str(thr) + '_nse' + str(noise)+'.wav'
writeResults(resultsFilename, azimuth_gt, elevation_gt, reverb, azMean, azDev, elMean, elDev, azMSE, elMSE, thr, noise)

#%%
filename = 'drums_FUMA_FUMA_r_True(-30, -29).wav';

azimuth_gt, elevation_gt, reverb = readGroundTruth(filename)
 #%%
thresholds = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
index=0
mse_results = np.zeros((np.size(thresholds), np.size(noise),2))
for thr in range (len(thresholds)):
    for nse in range (len(noise)):
        print(index)
        mse_results[thr,nse, :] = getDoaResults(filename,thresholds[thr],noise[nse], 1.04, -0.52)
        index = index +1
       
PlotMSEVariables(mse_results, thresholds, noise, 'Diffuseness Threshold', 'Noise Level', 'MSE',
                 'Azimuth MSE based on Diffuseness Thr.','Elevation MSE based on Diffuseness Thr.', 
                 'Azimuth MSE based on Noise Level','Elevation MSE based on Noise Level',
                 'MSE based on Diffuseness Thr. and Noise', 1.04, -0.52)
