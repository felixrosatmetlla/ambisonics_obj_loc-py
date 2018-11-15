# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import the needed libraries to read audios, use numpy arrays and plot
import soundfile as sf
import numpy as np
import math
import matplotlib.pyplot as plot

#Read audio file
data, samplerate = sf.read('271053__milanvdmeer__violinsingle-130-4mf-4.wav')

#Input variables by user
azimuth = math.pi*3/4
elevation = 0

amb_order = 1

norm = 'N3D'
ch_order = 'ACN'

#Ambisonics Order 1 formulas
W = data*1
X = data*math.cos(azimuth)*math.cos(elevation)
Y = data*math.sin(azimuth)*math.cos(elevation)
Z = data*math.sin(elevation)

#%% Normalization

#Get the number of channels the audio will have
n_channels = 0
for x in range(0,amb_order+1):
        n_channels = n_channels + (2*x + 1)
        
norm_fact = np.zeros(n_channels)

#Get the normalization factors based on the order, and the normalization desired by user
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
      
#Apply the normalization to the audio channels
W = W*norm_fact[0]   
Y = Y*norm_fact[1]
Z = Z*norm_fact[2]
X = X*norm_fact[3]
    
#Order the channels in the desired format
if ch_order=='FUMA':
    audio = np.vstack((W,X,Y,Z))

elif ch_order=='ACN':
    audio = np.vstack((W,Y,Z,X))

final_audio = np.transpose(audio)

#Write the audio file
sf.write('violinsingle_%s_%s(%d, %d).wav'%(norm,ch_order,azimuth*180/math.pi, elevation*180/math.pi),final_audio,samplerate)

#%% Plots

#Plot X and Y directions and gains without normalization
theta = np.arange(0,2*np.pi,0.1)
r_x = np.cos(theta)
r_y = np.sin(theta)

#Plot W gain without normalization
r_w = np.ones(theta.size)

#Get the normalized values
nr_w = r_w*norm_fact[0] 
nr_x = r_x*norm_fact[3]
nr_y = r_y*norm_fact[1]

#Create the subplots and plot
f, axarr = plot.subplots(2, 2, subplot_kw=dict(projection='polar'))


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

plot.show()