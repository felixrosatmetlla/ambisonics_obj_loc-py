# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import the needed libraries to read audios, use numpy arrays and plot
import soundfile as sf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import xml.etree.ElementTree as ET

#%%

def getAudioPath(filename):
    path = os.getcwd()
    path = os.path.dirname(path)
    path = os.path.join(path,'test/input/'+ filename) 
    return path

def getOutputAudioPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test/output/BFormat_Audios/'+ output_filename) 
    return output_path

def getGTPath(output_filename):
    output_path = os.getcwd()
    output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path,'test/output/Ground_Truth/'+ output_filename) 
    return output_path

def num_channels(amb_order):
    n_channels = 0
    for x in range(0,amb_order+1):
        n_channels = n_channels + (2*x + 1)
        
    return n_channels

def norm_factors(n_channels,amb_order,norm):
    norm_fact = np.zeros(n_channels)
    norm_id = 0
    
    for x in range (0,amb_order+1):
        for i in range (-x,x+1):
            if norm == 'N3D':
                if i == 0:
                    norm_fact[norm_id] = math.sqrt(2*x + 1)
                else:
                    norm_fact[norm_id] = math.sqrt(2*(2*x+1)*(math.factorial(x-abs(i))/math.factorial(x+abs(i))))
            
            elif norm == 'FUMA':
                if x == 0:
                    norm_fact[norm_id] = 1/math.sqrt(2)
                else:
                    norm_fact[norm_id] = 1
           
            elif norm == 'SN3D':
                if i == 0:
                    norm_fact[norm_id] = 1
                else:
                    norm_fact[norm_id] = math.sqrt(2*(math.factorial(x-abs(i))/math.factorial(x+abs(i))))
            
            norm_id += 1
            
    return norm_fact

def angleInterp(angle, time):
    x = time
    y = angle
    f = interp1d(x, y)
    newAngle = np.arange(0, 44100)
    interpAngle = f(newAngle)
    
    return interpAngle
    
def toAmbisonics(data,norm_fact,interpAzimuth, interpElevation):
    W = data*1*norm_fact[0]
    X = np.multiply(np.multiply(data,np.cos(interpAzimuth)),np.cos(interpElevation))*norm_fact[3]
    Y = np.multiply(np.multiply(data, np.sin(interpAzimuth)) , np.cos(interpElevation))*norm_fact[1]
    Z = np.multiply(data, np.sin(interpElevation))*norm_fact[2]
    
    return W,X,Y,Z

def toAmbisonicsMovReverb(data,norm_fact, interpAzimuth, interpElevation, reverbW):
    W = np.convolve(data, reverbW)*1*norm_fact[0]
    X = np.convolve(np.multiply(np.multiply(data,np.cos(interpAzimuth)),np.cos(interpElevation)),reverbW)*norm_fact[3]
    Y = np.convolve(np.multiply(np.multiply(data, np.sin(interpAzimuth)) , np.cos(interpElevation)),reverbW)*norm_fact[1]
    Z = np.convolve(np.multiply(data, np.sin(interpElevation)), reverbW)*norm_fact[2]
    
    return W,X,Y,Z
def order_channels(ch_order,W,X,Y,Z):
    if ch_order=='FUMA':
        audio = np.vstack((W,X,Y,Z))
    
    elif ch_order=='ACN':
        audio = np.vstack((W,Y,Z,X))
    
    audio = np.transpose(audio)
    return audio

def signal_gain(gain):
    gain_pos = np.empty(0)
    gain_neg = np.empty(0)
    
    for x in gain:
        if x >= 0:
            gain_pos = np.append(gain_pos,x)
            gain_neg = np.append(gain_neg,0)
        else:
            gain_neg = np.append(gain_neg,x)
            gain_pos = np.append(gain_pos,0)
    
    gain_neg = np.abs(gain_neg)
    return gain_pos, gain_neg

def groundTruth(azi, ele,filenm):
    
    data = ET.Element('data')
    title = ET.SubElement(data,'title')
    filename = ET.SubElement(data,'filename')
    azimuth = ET.SubElement(data, 'azimuth')
    elevation = ET.SubElement(data, 'elevation')
    
    title.set('name','GroundTruth')
    filename.set('name','Filename')
    azimuth.set('name','Azimuth')
    elevation.set('name','Elevation')
    
    title.text = 'Ground Truth'
    filename.text = filenm
    azimuth.text = str(azi)
    elevation.text = str(ele)
    
    file = filenm.split(".")
    path = getGTPath("groundTruth_" + file[0] + ".xml")
    groundtruth = ET.ElementTree(data)
    groundtruth.write(path)


#%% Input variables by user
rev = False

#azimuth = [np.pi/4,np.pi/2, 3*np.pi/4, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
azimuth = [np.pi,-np.pi/4,3*np.pi/4,-3*np.pi/4,5*np.pi/4,0]
elevation = [np.pi/4,-np.pi/2,np.pi/7,np.pi/3,-np.pi/2,0]
#elevation = [0,0,0,0,0,0]
time = [0, 8820, 17640, 26460, 35280, 44100]
amb_ord = 1

norm = 'FUMA'
ch_order = 'FUMA'
filename = 'drums.wav'

output_filename = 'drums_%s_%s_r_%s(%d, %d).wav'%(norm,ch_order,rev,azimuth[0]*180/math.pi, elevation[0]*180/math.pi)    

#%% Get Path and read audio file

path = getAudioPath(filename)
out_path = getOutputAudioPath(output_filename)

#Read audio file
data, samplerate = sf.read(path)

data = data[:samplerate,0];


#%% Normalization

#Get the number of channels the audio will have
n_ch = num_channels(amb_ord)

#Get the normalization factors based on the order, and the normalization desired by user
norm_fact = norm_factors(n_ch,amb_ord,norm)

#Interpolate
interpAzi = angleInterp(azimuth, time)
interpEle = angleInterp(elevation, time)

#%% Encode the audio to Ambisonics
rev_file_path = '/Users/felixrosatmetlla/Desktop/TFG/ambisonics_obj_loc-py/S3A/MainChurch/Soundfield/ls7.wav'

if rev == True:
    reverbData, revSamplerate = sf.read(rev_file_path)
    reverbW = reverbData[:,0]

    W,X,Y,Z = toAmbisonicsMovReverb(data,norm_fact, interpAzi, interpEle,reverbW)

elif rev == False:
    W,X,Y,Z = toAmbisonics(data,norm_fact, interpAzi, interpEle)

#%%
    
#Order the channels in the desired format      
audio = order_channels(ch_order,W,X,Y,Z)

#Write the audio file
sf.write(out_path,audio,samplerate)

#Write Ground Truth file
groundTruth(azimuth, elevation, output_filename)

#%% Plots   

azi = np.arange(0,2*np.pi,0.01)

w_gains = np.ones(azi.size)*norm_fact[0]
x_gains = np.cos(azi)*norm_fact[3]
y_gains = np.sin(azi)*norm_fact[1]

xg_pos, xg_neg = signal_gain(x_gains)
yg_pos, yg_neg = signal_gain(y_gains)

plt.figure(0)
plt.plot(azi,w_gains)

plt.figure(1)
plt.plot(azi,x_gains)

plt.figure(2)
plt.plot(azi,y_gains)

plt.figure(3)
plt.polar(azi,w_gains)

plt.figure(4)
plt.polar(azi,xg_pos)
plt.polar(azi,xg_neg)

plt.figure(5)
plt.polar(azi,yg_pos)
plt.polar(azi,yg_neg)

plt.show()  