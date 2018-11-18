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
import os

#%%

def getAudioPath(filename):
    path = os.getcwd()
    path = os.path.dirname(path)
    path = os.path.join(path,'test_audios/'+ filename) 
    return path

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

def toAmbisonics(data,norm_fact):
    W = data*1*norm_fact[0]
    X = data*math.cos(azimuth)*math.cos(elevation)*norm_fact[3]
    Y = data*math.sin(azimuth)*math.cos(elevation)*norm_fact[1]
    Z = data*math.sin(elevation)*norm_fact[2]
    
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

#%% Input variables by user

azimuth = math.pi*3/4
elevation = 0

amb_ord = 1

norm = 'N3D'
ch_order = 'ACN'
filename = '271053__milanvdmeer__violinsingle-130-4mf-4.wav'

#%% Get Path and read audio file

path = getAudioPath(filename)

#Read audio file
data, samplerate = sf.read(path)

#%% Normalization

#Get the number of channels the audio will have
n_ch = num_channels(amb_ord)

#Get the normalization factors based on the order, and the normalization desired by user
norm_fact = norm_factors(n_ch,amb_ord,norm)
      
#Apply the normalization to the audio channels
W,X,Y,Z = toAmbisonics(data,norm_fact)
    
#Order the channels in the desired format      
audio = order_channels(ch_order,W,X,Y,Z)

#Write the audio file
sf.write('violinsingle_%s_%s(%d, %d).wav'%(norm,ch_order,azimuth*180/math.pi, elevation*180/math.pi),audio,samplerate)

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