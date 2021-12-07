#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:38:57 2021

@author: nejc_novak
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.animation as animation
%matplotlib qt5


#parametri za generiranje sinusa
px = 10 
št_period = 1
širina_območja = 10 #[mm]
dolžina = 0.1 #[s]
fps = 100

število_sličic = int((dolžina*fps))
x, h = np.linspace(0,širina_območja,px, retstep=True)
t = np.linspace(0, dolžina, število_sličic-1)


#premikanje sinusa
pravi_pomik = 0.2 #[mm]
I = np.zeros([px, število_sličic])
for i in range(0, število_sličic):
    I[:,i] += (np.sin(2*np.pi*(x-i*pravi_pomik)/(širina_območja/št_period))+1)/2



#delta I
delta_I = np.zeros([px,število_sličic-1])
for i in range(0,število_sličic-1):
        delta_I[:,i] += I[:,i+1] - I[:,i]
        

#izračun pomikov   
odvod_x, odvod_t = np.gradient(I,h, edge_order=2)
delta_x = - delta_I / odvod_x[:,:-1]
povprečje = sum(delta_x[:,0]) / len(delta_x[:,0])
napaka = abs(pravi_pomik-povprečje)
relativno = napaka / pravi_pomik * 100
print(f'Napaka pomika iz vhodnih podatkov {napaka:.6f}mm.')
print(f'Relativna napaka pomika iz vhodnih podatkov {relativno:.2f}%.\n')


# interpoliranje s spline
px2 = 500
x2, h2 = np.linspace(0,širina_območja,px2, retstep=True)

I2 = np.zeros([px2, število_sličic])
for i in range(0, število_sličic):
    interp_spline = UnivariateSpline(x, I[:,i])
    interp_spline.set_smoothing_factor(0)
    I2[:,i] += interp_spline(x2)
    
    
#delta I - izboljšani podatki
delta_I2 = np.zeros([px2,število_sličic-1])
for i in range(0,število_sličic-1):
        delta_I2[:,i] += I2[:,i+1] - I2[:,i] 


#Izračun pomikov - izboljšani podatki
odvod_x2, odvod_t2 = np.gradient(I2,h2, edge_order=2)
delta_x2 = - delta_I2 / odvod_x2[:,:-1]
povprečje2 = sum(delta_x2[:,0]) / len(delta_x2[:,0])
napaka2 = abs(pravi_pomik-povprečje2)
relativno2 = napaka2 / pravi_pomik * 100
print(f'Napaka pomika iz popravljenih podatkov {napaka2:.6f}mm.')
print(f'Relativna napaka pomika iz popravljenih podatkov {relativno2:.2f}%.')

#generična vrstica
vrstica = np.sin(2*np.pi*x)

#odvodi
odvod1 = np.gradient(vrstica, edge_order=2)
odvod2 = np.gradient(vrstica, h, edge_order=2)
odvod3 = 2*np.pi*np.cos(2*np.pi*x)

#Različni načini numeričnega odvajanja
# plt.plot(x, vrstica, label='vhodni podatki - funkcija')
# plt.plot(x, odvod1, label='numerično odvajanje brez koraka')
# plt.plot(x, odvod2, label='numerično odvajanje s korakom')
# plt.plot(x, odvod3, label='matematični odvod')
# plt.title('Različni načini odvajanja')
# plt.legend()
# plt.show()

#Sintetični premik sinusne krivulje
# plt.plot(x, I[:,0],label='prva sličica')
# plt.plot(x, I[:,1],label='druga sličica')
# plt.plot(x, I[:,2],label='tretja sličica')
# # plt.title('Sintetični premik')
# plt.ylabel('I [/]')
# plt.xlabel('x [mm]')
# plt.legend()
# plt.show()

#Prikaz izboljšanih podatkov
# plt.plot(x, I[:,0], label='prva sličica, vhodni podatki')
# plt.plot(x2, I2[:,0], label='prva sličica, izboljšani podatki')
# # plt.title('Interpolirani podatki')
# plt.ylabel('I [/]')
# plt.xlabel('x [mm]')
# plt.legend()
# plt.show()

#Prikaz pomikov po sliki - vhodni podatki
# plt.plot(x, delta_x[:,0], label='pomik iz 1. sličice')
# plt.plot(x, delta_x[:,1], label='pomik iz 2. sličice')
# plt.plot(x, delta_x[:,2], label='pomik iz 3. sličice')
# plt.axhline(y=povprečje, color='r', linestyle='--', label='povprečje')
# plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
# plt.title('Pomik izračunan iz prvotnih podatkov')
# plt.ylabel('pomik [mm]')
# plt.xlabel('x [mm]')
# plt.legend()
# plt.show()

#Prikaz pomikov po sliki - izboljšani podatki
# plt.plot(x2, delta_x2[:,0], label='pomik iz 1. sličice')
# plt.plot(x2, delta_x2[:,1], label='pomik iz 2. sličice')
# plt.plot(x2, delta_x2[:,2], label='pomik iz 3. sličice')
# plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
# plt.axhline(y=povprečje2, color='r', linestyle=':', label='povprečje')
# plt.title('Pomik izračunan iz interpoliranih podatkov')
# plt.ylabel('pomik [mm]')
# plt.xlabel('x [mm]')
# plt.legend()
# plt.show()


#Prikaz pomikov skozi čas - vhodni podatki
# srednji_px = int(px/2)
# plt.plot(t, delta_x[1,:], label='pomik iz 1. piksla')
# plt.plot(t, delta_x[srednji_px,:], label='pomik iz sredinskega piksla')
# plt.plot(t, delta_x[-1,:], label='pomik iz zadnjega piksla')
# plt.axhline(y=povprečje, color='r', linestyle=':', label='povprečje')
# plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
# # plt.title('Pomik v času - vhodni podatki')
# plt.ylabel('pomik [mm]')
# plt.xlabel('t [s]')
# plt.legend()
# plt.show()


#Prikaz pomikov skozi čas - izboljšani podatki
# srednji_px2 = int(px2/2)
# plt.plot(t, delta_x2[0,:], label='pomik iz 1. piksla')
# plt.plot(t, delta_x2[srednji_px2,:], label='pomik iz sredinskega sličice')
# plt.plot(t, delta_x2[-1,:], label='pomik iz zadnjega piksla')
# plt.axhline(y=povprečje2, color='r', linestyle=':', label='povprečje')
# plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
# plt.title('Pomik v času - interpolirani podatki')
# plt.ylabel('pomik [mm]')
# plt.xlabel('t [s]')
# plt.legend()
# plt.show()

#Prikaz pomikov skozi čas - skupaj
# plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
# plt.axhline(y=povprečje, color='orange', linestyle=':', label='povprečje - vhodni p.')
# plt.axhline(y=povprečje2, color='b', linestyle=':', label='povprečje - interpolirani p.')
# plt.plot(t, delta_x[5,:], color='orange', label='pomik iz 1. sličice - vhodni p.')
# plt.plot(t, delta_x2[5,:], color='b', label='pomik iz 1. sličice - interpolirani p.')
# # plt.title('Pomik v času - primerjava')
# plt.ylabel('pomik [mm]')
# plt.xlabel('t [s]')
# plt.legend()
# plt.show()

#Prikaz težve gradientne metode - vhodni podatki
# plt.plot(x, delta_x[:,0], label='pomik iz 1. sličice')
# plt.plot(x, I[:,0], label='intenziteta po x', linestyle=':', alpha=0.8)
# plt.axhline(y=povprečje, color='r', linestyle='--', label='povprečje')
# plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
# # plt.title('Prikaz težave gradiente metode')
# plt.ylabel('pomik [mm]')
# plt.xlabel('x [mm]')
# plt.legend()
# plt.show()

#Prikaz težve gradientne metode - izboljšani podatki
# plt.plot(x2, delta_x2[:,0], label='pomik iz 1. sličice')
# plt.plot(x2, I2[:,0], label='intenziteta po x',linestyle=':')
# plt.plot(x2, odvod_x2[:,0], label='gradient', linestyle=':')
# # plt.title('Prikaz težave gradientne metode')
# plt.ylabel('pomik [mm]')
# plt.xlabel('x [mm]')
# plt.legend()
# plt.grid(axis='y')
# plt.show()



    