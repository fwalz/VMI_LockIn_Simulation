# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:15:27 2025

@author: ShivaramLab
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.fftpack import fft, ifft, fftfreq

# Define the parameters of the square wave
frequency = 1000  # 1kHz
sampling_rate = 1000  # Sampling rate
duration = 0.1  # Duration of the signal in seconds


#####Lock In Code#####


############################## Parameters setting ###############################
# True Signal
r = 1 #Signal Amplitude
phi = np.pi/6 #Signal Phase

# Lock-in paremeters
freq = 100
w = freq*2*np.pi #Lock-in/ Chopper angular freq. for f = 200 Hz

tf = 1 #Exhibiting time window (sec) 
dt = 1e-3  #Time resolution (sec); pulse repetition rate ~ 1000 Hz
N = int(tf/dt) #number of points; Exhibiting time window

T = 1 #Integration time (sec)
n = int(T/dt) #number of points; Integration

print('Total sampling time window =',tf+T, 's')
print('Freq =',w/(2*np.pi), 'Hz')
print('Time resolution =',tf/N*1e3, 'ms')
print('Integration time =',w*T/(2*np.pi),'periods')

#(Unphysical) Variables initailization 
X = np.zeros(N)
Y = np.zeros(N)
sig_out = np.zeros(N)
sum = 0
sum2 = 0
#################################################################################

############################## Main #############################################
t = np.linspace(0, tf+T, N+n, endpoint=False) #Sampling time window
# Generate the square wave that is only above 0
square_wave_LI = (signal.square(w * t) +1)/2
# Generate the square wave that is only above -1
#square_wave_LI = signal.square(w * t)
##################
hit_num = 10 #true signal value
dist = poisson.rvs(mu=0, size=int(N+n)) #noise in hit value
#dist = 0
sig = dist + hit_num

#multiply by the suare wave at the lock in frequency
sig = sig*square_wave_LI

#noise in general, maybe it can be white or pink or poission
noise = poisson.rvs(mu=0, size=int(N+n))
noise_white = np.random.normal(0,0, N+n) #Gaussian(mean, STDV, size)
sig_in = sig + noise + noise_white

###################################################################
############################ Mixer ################################
# In-phase mixing###
mixer1 = np.cos(w*t)
mix1 = sig_in*mixer1

# Qradurature mixing#
mixer2 = -np.sin(w*t)
mix2 = sig_in*mixer2
###################################################################
################ Integration (taking time avg) ####################
for i in range(n):
    sum += mix1[i]*dt/T
    sum2 += mix2[i]*dt/T
for i in range(N):
    X[i] = sum
    Y[i] = sum2
    sum = sum - mix1[i]*dt/T + mix1[i+n]*dt/T
    sum2 = sum2 - mix2[i]*dt/T + mix2[i+n]*dt/T
###################################################################
####################### Signal recovering #########################
theta = np.arctan(Y/X)
R = np.sqrt(Y**2+X**2)

#Adding the correction for the frequency splitting 
R = R *(np.pi)

for i in range (N):
    sig_out[i] = R[i]*np.cos(w*t[i]+theta[i])
###################################################################
####################### SNR calculation ###########################
# Initial SNR 
STDV = 0
for i in range(N):
    STDV += (sig_in[i]-sig[i])**2
    
STDV = np.sqrt(STDV/N)
print('SNR(Amp.)_before =',abs(r/STDV))

# Filtered SNR
STDV = 0
for i in range(N):
    STDV += (sig_out[i]-sig[i])**2

STDV = np.sqrt(STDV/N)
print('SNR(Amp.)_after =', abs(r/STDV))
###################################################################
#################################################################################

################# Ploting parameters_ no physics here ###########################
freq = fftfreq(N+n,dt)

plt.figure(figsize=(8,6))
plt.title("Noise spectrum", size = 20)
plt.plot(freq[:(N+n)//2], abs(fft(sig))[:(N+n)//2] ,label = "$data$", c = 'black') #positive freq. component
#plt.plot(freq[:(N+n)//2], 10*np.log10(abs(fft(sig)))[:(N+n)//2] ,label = "$data$", c = 'black') #positive freq. component in db
#plt.plot(freq[(N+n)//2+1:], 10*np.log10(abs(ft_noise))[(N+n)//2+1:] ,label = "$data$", c = 'black') #negative freq. component
plt.xticks(size = 12)
plt.yticks(size = 12)
#plt.xscale("log")
plt.xlabel("Frequency (Hz)",size = 20) 
plt.ylabel("Spectral Amplitude (dB, a.u.)",size=20) 
#plt.legend(loc='lower right',fontsize=20)
plt.show() 

print("max of signal is "+str(np.max(sig_out)))

plt.figure(figsize=(8.5,6))
plt.title("Signal buried in noise", size = 20)
plt.plot(t, sig_in ,label = "$data$", c = 'black')
plt.xlim(0,tf/2)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time (s)",size = 20) 
plt.ylabel("Amplitude (a.u.)",size=20) 
#plt.legend(loc='lower right',fontsize=20)
plt.show() 

plt.figure(figsize=(8,6))
plt.title("Output Signal", size = 20)
plt.plot(t[:N],sig_out ,label = "Filtered Sig", c = 'black')
plt.plot(t[:N],sig[:N] ,label = "Real Sig", c = 'grey', ls ='--')
plt.xlim(0,tf/2)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time (s)",size = 20) 
plt.ylabel("Amplitude (a.u.)",size=20) 
plt.legend(loc='lower right',fontsize=18)
plt.show() 


#Plot of the signal before noise
plt.figure(figsize=(8.5,6))
plt.title("Signal with no noise", size = 20)
plt.plot(t, sig ,label = "$data$", c = 'black')
plt.xlim(0,tf/2)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time (s)",size = 20) 
plt.ylabel("Amplitude (a.u.)",size=20) 
#plt.legend(loc='lower right',fontsize=20)
plt.show() 
"""
freq = fftfreq(N+n,dt)
plt.figure(figsize=(8,6))
plt.title("Noise spectrum", size = 20)
plt.plot(freq[:(N+n)//2], 10*np.log10(abs(fft(sig_in-sig)))[:(N+n)//2] ,label = "$data$", c = 'black') #positive freq. component
#plt.plot(freq[(N+n)//2+1:], 10*np.log10(abs(ft_noise))[(N+n)//2+1:] ,label = "$data$", c = 'black') #negative freq. component
plt.xticks(size = 12)
plt.yticks(size = 12)
#plt.xscale("log")
plt.xlabel("Frequency (Hz)",size = 20) 
plt.ylabel("Spectral Amplitude (dB, a.u.)",size=20) 
#plt.legend(loc='lower right',fontsize=20)
plt.show() 
"""
"""
plt.figure(figsize=(8.5,6))
plt.title("Signal after Mixer1 (in-phase)", size = 20)
plt.plot(t, mix1 ,label = "$Mix1$", c = 'b')
plt.xlim(0,8*2*np.pi/w)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time (s)",size = 20) 
plt.ylabel("Amplitude (a.u.)",size=20) 
plt.legend(loc='lower right',fontsize=20)
plt.show() 

plt.figure(figsize=(8.5,6))
plt.title("Signal after Mixer2 (quadrature)", size = 20)
plt.plot(t, mix2 ,label = "$Mix2$", c = 'r')
plt.xlim(0,8*2*np.pi/w)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time (s)",size = 20) 
plt.ylabel("Amplitude (a.u.)",size=20) 
plt.legend(loc='lower right',fontsize=20)
plt.show() 
"""

"""
plt.figure(figsize=(8,6))
plt.title("In-phase componet X", size = 20)
plt.plot(t[:N], X ,label = "$X_{avg}$", c = 'b')
plt.xlim(0,8*2*np.pi/w)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time (s)",size = 20) 
plt.ylabel("Amplitude (a.u.)",size=20) 
plt.legend(loc='lower right',fontsize=20)
plt.show() 

plt.figure(figsize=(8,6))
plt.title("Quadrature componet Y", size = 20)
plt.plot(t[:N], Y ,label = "$Y_{avg}$", c = 'r')
plt.xlim(0,8*2*np.pi/w)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.xlabel("Time(s)",size = 20) 
plt.ylabel("Amplitude(a.u.)",size=20) 
plt.legend(loc='lower right',fontsize=20)
plt.show() 
"""
#################################################################################
################################# End ###########################################