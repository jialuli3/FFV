import scipy.io.wavfile as wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
Tframe=0.032
Tskip=0.01
T0=0.004
T1=0.012
Tfb=0.0013
N=512

def processWav(inputWavFile):
    """
    rate(int): wave processing rate
    frames(numpy array): array of extract frames
    """
    rate, data=wavfile.read(inputWavFile)
    skip=int(Tskip*rate)
    dur=int(Tframe*rate)
    processedData=np.asarray(copy.deepcopy(data),dtype=np.float64)
    #processedData=np.asarray(copy.deepcopy(data),dtype=np.int16)/np.power(2,15)
    #plotSpectrogram(data,rate)
    #plotGraph(processedData,"indexes","signals","Speech Signals")
    processedData-=np.mean(processedData)
    energy=np.linalg.norm(processedData)
    processedData/=energy
    frames=np.zeros((int((data.shape[0]-dur)/skip),dur))
    nframes=int((data.shape[0]-dur)/skip)
    for i in range(nframes):
        if i==nframes-1:
            data_point=len(processedData[i*skip:i*skip+dur])
            frames[i,:data_point]=processedData[i*skip:i*skip+dur]
        else:
            frames[i]=processedData[i*skip:i*skip+dur]
    #print(rate,frames.shape)
    return rate,frames

def computeWin(rate):
    """
    Return left and right window
    """
    hamming=np.hamming(int(2*T1*rate))[:int(T1*rate)]
    hanning=np.hanning(int(4*T0*rate))[int(2*T0*rate):]
    zeros=np.zeros(int(T1*rate))
    window=np.concatenate((hamming,hanning,zeros),axis=0)
    return window,window[::-1]

def computefft(frames,win):
    fftFrame=np.zeros_like(frames,dtype=np.complex_)
    for i in range(len(frames)):
        wframe=frames[i]*win
        fftFrame[i]=np.fft.fft(wframe,n=N)
    return fftFrame

def computeIpFrame(r,frame):
    #r=np.linspace(-N//2+1,N/2+1,N)
    k=np.linspace(-N//2+1,N//2,N)
    rho=-abs(4*r/N)
    idx=np.power(2,rho)*k
    alpha=np.absolute(np.ceil(idx)-idx)
    return alpha*frame[np.asarray(np.floor(idx)+N//2-1,dtype=np.int16)]\
        +(1-alpha)*frame[np.asarray(np.ceil(idx)+N//2-1,dtype=np.int16)]

def computeffv(frames,rate,testing_frame=2):
    g=np.zeros((testing_frame,N))
    left_win,right_win=computeWin(rate)
    F_L=computefft(frames,left_win)
    F_R=computefft(frames,right_win)
    for i in range(testing_frame):
        #r<0
        for r in range(-N//2+1,N//2+1):
            g_idx=r+N//2-1
            if r<0:
                F_Rip=computeIpFrame(r,F_R[i])
                g[i,g_idx]=np.sum(np.absolute(np.conj(F_Rip))*np.absolute(F_L[i,:]))/\
                        np.sqrt(np.sum(np.square(np.absolute(np.conj(F_Rip))))*np.sum(np.square(np.absolute(F_L[i,:]))))
            else:
                F_Lip=computeIpFrame(r,F_L[i])
                g[i,g_idx]=np.sum(np.absolute(np.conj(F_R[i,:]))*np.absolute(F_Lip))/\
                        np.sqrt(np.sum(np.square(np.absolute(np.conj(F_R[i,:]))))*np.sum(np.square(np.absolute(F_Lip))))
    #for i in range(len(frames)):
    # for i in range(testing_frame):
    #     for r in range(-N//2+1,1): #r<0
    #         F_Rip_sum=0
    #         g_idx=r+N//2-1
    #         for k in range(-N//2+1,N//2+1):
    #             F_Lidx=k+N//2-1
    #             F_Rip=computeIpFrame(F_R[i],r,k)
    #             g[i,g_idx]+=np.absolute(F_Rip)*np.absolute(np.conj(F_L[i,F_Lidx]))
    #             F_Rip_sum+=np.square(np.absolute(F_Rip))
    #         g[i,g_idx]/=np.sqrt(np.sum(F_Rip_sum)*np.sum(np.square(np.absolute(np.conj(F_L[i,:])))))
    #     for r in range(1,N//2+1): #r>=0
    #         F_Lip_sum=0
    #         g_idx=r+N//2-1
    #         for k in range(-N//2+1,N//2+1):
    #             F_Ridx=k+N//2-1
    #             F_Lip=computeIpFrame(F_R[i],r,k)
    #             g[i,g_idx]+=np.absolute(F_Lip)*np.absolute(np.conj(F_R[i,F_Ridx]))
    #             F_Lip_sum+=np.square(np.absolute(F_Lip))
    #         g[i,g_idx]/=np.sqrt(np.sum(F_Lip_sum)*np.sum(np.square(np.absolute(np.conj(F_R[i,:])))))
    #plotGraph(g[testing_frame-1,:])
    return g

def computeFilterBank(rate,numFB=7):
    frame_len=int(Tframe*rate)
    fb_int_len=int(Tfb*rate)
    filterBanks=np.zeros((numFB,frame_len))
    tall_trap_height=0.25
    short_trap_height=0.16
    tall_trap=np.linspace(0,tall_trap_height,fb_int_len)
    short_trap=np.linspace(0,short_trap_height,fb_int_len*2)
    #first filterbank
    filterBanks[0,-3*fb_int_len+N//2-1:-2*fb_int_len+N//2-1]=tall_trap
    filterBanks[0,2*fb_int_len+N//2-1:3*fb_int_len+N//2-1]=tall_trap[::-1]
    filterBanks[0,-2*fb_int_len+N//2-1:2*fb_int_len+N//2-1]=tall_trap_height
    #second filterbank
    filterBanks[1,N//2-1:2*fb_int_len+N//2-1]=short_trap
    filterBanks[1,6*fb_int_len+N//2-1:8*fb_int_len+N//2-1]=short_trap[::-1]
    filterBanks[1,2*fb_int_len+N//2-1:6*fb_int_len+N//2-1]=short_trap_height

    filterBanks[2,-2*fb_int_len+N//2-1:N//2-1]=short_trap[::-1]
    filterBanks[2,-8*fb_int_len+N//2-1:-6*fb_int_len+N//2-1]=short_trap
    filterBanks[2,-6*fb_int_len+N//2-1:-2*fb_int_len+N//2-1]=short_trap_height

    filterBanks[3,4*fb_int_len+N//2-1:6*fb_int_len+N//2-1]=short_trap
    filterBanks[3,10*fb_int_len+N//2-1:12*fb_int_len+N//2-1]=short_trap[::-1]
    filterBanks[3,6*fb_int_len+N//2-1:10*fb_int_len+N//2-1]=short_trap_height

    filterBanks[4,-6*fb_int_len+N//2-1:-4*fb_int_len+N//2-1]=short_trap[::-1]
    filterBanks[4,-12*fb_int_len+N//2-1:-10*fb_int_len+N//2-1]=short_trap
    filterBanks[4,-10*fb_int_len+N//2-1:-6*fb_int_len+N//2-1]=short_trap_height

    filterBanks[5,8*fb_int_len+N//2-1:10*fb_int_len+N//2-1]=short_trap
    filterBanks[5,10*fb_int_len+N//2-1:N]=short_trap_height

    filterBanks[6,-10*fb_int_len+N//2-1:-8*fb_int_len+N//2-1]=short_trap[::-1]
    filterBanks[6,-N:-10*fb_int_len+N//2-1]=short_trap_height

    # for i in range(numFB):
    #    plt.plot(filterBanks[i,:])
    # plt.show()
    return filterBanks.T

def computeffvfb(filterBanks,frames):
    numFB=filterBanks.shape[1]
    ffvFB=np.zeros((len(frames),numFB))
    for i in range(len(frames)):
        ffvFB[i,:]=np.sum(filterBanks*frames[i][:,None],axis=0)
    print(ffvFB)
    return ffvFB

def plotSpectrogram(data,rate,title="Original speech spectrogram"):
    f,t,Sxx=signal.spectrogram(data,rate,nfft=N)
    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            Sxx[i][j]=np.where(Sxx[i][j]>0.01*np.max(Sxx[i]),Sxx[i][j],0)
        Sxx[i]/=np.max(Sxx[i])
    plt.pcolormesh(t,f,Sxx,cmap=plt.cm.get_cmap('Blues_r'))
    plt.ylabel("frequency [Hz]")
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.show()

def plotGraph(graph,x_label="",y_label="",title=""):
    plt.plot(graph)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

rate,frames=processWav("mk970009.wav")
g=computeffv(frames,rate,testing_frame=2)
filterBanks=computeFilterBank(rate)
computeffvfb(filterBanks,g)
