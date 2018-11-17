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

filename = '271053__milanvdmeer__violinsingle-130-4mf-4.wav'

path = os.getcwd()
path = os.path.dirname(path)
path = os.path.join(path,'test_audios/271053__milanvdmeer__violinsingle-130-4mf-4.wav')
print(path)


#%%
#Read audio file
data, samplerate = sf.read(path)

#Input variables by user
azimuth = math.pi*3/4
elevation = 0

amb_ord = 1

norm = 'N3D'
ch_order = 'ACN'

#Ambisonics Order 1 formulas
W = data*1
X = data*math.cos(azimuth)*math.cos(elevation)
Y = data*math.sin(azimuth)*math.cos(elevation)
Z = data*math.sin(elevation)

#%% Normalization
def num_channels(amb_order):
    n_channels = 0
    for x in range(0,amb_order+1):
        n_channels = n_channels + (2*x + 1)
        
    return n_channels


#Get the number of channels the audio will have
n_ch = num_channels(amb_ord)

def norm_factors(n_channels,amb_order,norm):
    norm_fact = np.zeros(n_channels)
    norm_id = 0
    
    for x in range (0,amb_order+1):
        for i in range (-x,x+1):
            if norm == 'N3D':
                if i == 0:
                    norm_fact[norm_id] = math.sqrt(2*x + 1)
                else:
                    norm_fact[norm_id] = math.sqrt(2*(2*x+1)*(math.factorial(x-i)/math.factorial(x+i)))
            
            elif norm == 'FUMA':
                if x == 0:
                    norm_fact[norm_id] = 1/math.sqrt(2)
                else:
                    norm_fact[norm_id] = 1
           
            elif norm == 'SN3D':
                if i == 0:
                    norm_fact[norm_id] = 1
                else:
                    norm_fact[norm_id] = math.sqrt(2*(math.factorial(x-i)/math.factorial(x+i)))
            
            norm_id += 1
            
    return norm_fact
      


#Get the normalization factors based on the order, and the normalization desired by user
normalization_fact = norm_factors(n_ch,amb_ord,norm)
      
#Apply the normalization to the audio channels
W = W*normalization_fact[0]   
Y = Y*normalization_fact[1]
Z = Z*normalization_fact[2]
X = X*normalization_fact[3]
    
#Order the channels in the desired format

def order_channels(ch_order,W,X,Y,Z):
    if ch_order=='FUMA':
        audio = np.vstack((W,X,Y,Z))
    
    elif ch_order=='ACN':
        audio = np.vstack((W,Y,Z,X))
    
    audio = np.transpose(audio)
    return audio

        
audio = order_channels(ch_order,W,X,Y,Z)


#Write the audio file
sf.write('violinsingle_%s_%s(%d, %d).wav'%(norm,ch_order,azimuth*180/math.pi, elevation*180/math.pi),audio,samplerate)

#%% Plots

#Plot X and Y directions and gains without normalization
theta = np.arange(0,2*np.pi,0.01)
r_x = np.cos(theta)
r_y = np.sin(theta)

#Plot W gain without normalization
r_w = np.ones(theta.size)

#Get the normalized values
nr_w = r_w*normalization_fact[0] 
nr_x = r_x*normalization_fact[3]
nr_y = r_y*normalization_fact[1]

#Create the subplots and plot
f, axarr = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))

#Plot channels W,X,Y
axarr[0, 0].plot(theta,r_w)
axarr[0, 0].set_theta_offset(math.pi*0.5)

axarr[0, 1].plot(theta,r_x*math.cos(azimuth))
axarr[0, 1].plot(theta,r_y*math.sin(azimuth))
axarr[0, 1].set_theta_offset(math.pi*0.5)
axarr[0, 1].set_rmax(1)
axarr[0, 1].set_rmin(0)

#Plot normalized channels W,X,Y
axarr[1, 0].plot(theta,nr_w)
axarr[1, 0].set_theta_offset(math.pi*0.5)

axarr[1, 1].plot(theta,nr_x*math.cos(azimuth))
axarr[1, 1].plot(theta,nr_y*math.sin(azimuth))
axarr[1, 1].set_rmin(0)
axarr[1, 1].set_theta_offset(math.pi*0.5)


plt.show()

#%%    

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


azi = np.arange(0,2*np.pi,0.01)

s = np.cos(azi)
k = np.sin(azi)

s_pos, s_neg = signal_gain(s)

k_pos, k_neg = signal_gain(k)

#plt.polar(azi,np.cos(azi))

plt.figure(0)
plt.plot(azi,s)

#plt.figure(1)
#plt.plot(azi,k)

plt.figure(2)
plt.polar(azi,s_pos)
plt.polar(azi,s_neg)

#plt.figure(3)

#plt.polar(azi,k_pos)
#plt.polar(azi,np.abs(k_neg))

    