#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:38:57 2021

@author: nejc_novak
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
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
pravi_pomik = 3 #[mm]
I = np.zeros([px, število_sličic])
for i in range(0, število_sličic):
    I[:,i] += (np.sin(2*np.pi*(x-i*pravi_pomik)/(širina_območja/št_period))+1)/2

točka_izračuna = 5
sličica = 5
število_iteracij = 5

#odvodi
odvod_x, odvod_t = np.gradient(I,h, edge_order=2)
odvod = odvod_x[:,sličica]

spline1 = UnivariateSpline(x, I[:,sličica-1], s=0)
spline2 = UnivariateSpline(x, I[:,sličica], s=0)


#iterativni izračun pomikov
pomik = 0
pomiki_iteracije = []
for i in range(0, število_iteracij):

    I_1 = spline1(x-pomik)
    I_2 = spline2(x)
    
    delta_I = I_2[sličica] - I_1[sličica]
    delta_x = - delta_I / odvod[točka_izračuna]
    pomik += delta_x
    pomiki_iteracije.append(pomik)
    napaka = abs(pravi_pomik-pomik)
    relativna = napaka / pravi_pomik * 100

    
    print(f'Pomik v {i+1}. iteraciji {pomik:.6f}mm.')
    print(f'Napaka pomika iz vhodnih podatkov {napaka:.4f}mm.')
    print(f'Relativna napaka pomika iz vhodnih podatkov {relativna:.2f}%.\n')



#prikaz lokacije izračuna
plt.plot(x, I[:,sličica-1], label='prva sličica')
plt.plot(x, I[:,sličica], label='druga sličica')
plt.axvline(x=točka_izračuna, color='r', linestyle='--', label='točka izračuna')
plt.title('Iterativni izračun')
plt.ylabel('I')
plt.xlabel('Px')
plt.legend()
plt.show()


#prikaz iteracij
plt.scatter(np.arange(1,število_iteracij+1), pomiki_iteracije, label='pomiki')
plt.axhline(y=pravi_pomik, color='g', linestyle='--', label='pravi pomik')
plt.locator_params(axis="x", integer=True)
plt.title('Izračunan pomik za vsako iteracijo')
plt.ylabel('pomik [mm]')
plt.xlabel('iteracije')
plt.legend()
plt.show()

    