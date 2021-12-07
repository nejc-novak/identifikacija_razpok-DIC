# -*- coding: utf-8 -*-
"""
@author: nejcn
"""

import pyidi
import pyMRAW
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import ticker, cm
# from scipy.fftpack import fft, fftfreq
import matplotlib.animation as animation
%matplotlib qt5

class Crack():
    """ Razred za uvažanje in preračun podatkov, ki smo jih zajeli.

    Parametri:
    pot : pot in ime datoteke s podatki
    """
    def __init__(self, path, name):
        self.file = path
        self.name = name
        self.video = pyidi.pyIDI(self.file)
        self.mraw, info = pyMRAW.load_video(self.file)
        self.frame_num = info["Total Frame"]
        self.frame_rate = info["Record Rate(fps)"]
    
    
    
    def automatic_points(self, ROI_size=10):
        """ Avtomatsko zgenerira točke znotraj izbranega območja.

            Funkcija vrne točke.

            Argumenti:
            ROI_size: velikost območja zanimanja
        """
        points_obj = pyidi.tools.GridOfROI(self, roi_size=(ROI_size, ROI_size), noverlap=0)
        points = points_obj.points
        self.video.set_points(points)
        #points_num = len(points)
        return(points)
           

            
    def show_points(self, points, frame=0):  
        """ Grafični prikaz točk.

            Argumenti:
            points: točke, ki jih želiš prikazat
        """
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(self.mraw[frame].astype(float), 'gray')
        ax.scatter(points[:, 1], points[:, 0], marker='.', color='b')
        ax.axhline(y=self.h_line1, color='r', linestyle='-', label='povprečje')
        ax.axhline(y=self.h_line2, color='r', linestyle='-', label='povprečje')
        ax.axvline(x=self.v_line1, color='r', linestyle='-', label='povprečje')
        ax.axvline(x=self.v_line2, color='r', linestyle='-', label='povprečje')
        plt.grid(False)
        plt.show()
    
    

    def rct_points(self, h_line1=180, h_line2=250, v_line1=220, v_line2=420, ROI_size=10):
        """ Zgenerira točke znotaj pravokotnika.

            Funkcija vrne x in y koordinate.

            Argumenti:
            h_line-x : linija, ki omejuje višino pravokotnika
            h_line-y : linija, ki omejuje širino pravokotnika
        """
        self.h_line1 = h_line1
        self.h_line2 = h_line2
        
        self.v_line1 = v_line1
        self.v_line2 = v_line2
        
        self.ROI = ROI_size           
        
        points_num_x = int(abs(v_line1-v_line2)/ROI_size) + 1 #(enak y)
        self.points_num_x = points_num_x
        points_x = np.linspace(v_line1, v_line2, points_num_x)
        self.points_x = points_x
        
        points_num_y = int(abs(h_line1-h_line2)/ROI_size) + 1# (enak x)
        self.points_num_y = points_num_y
        points_y = np.linspace(h_line1, h_line2, points_num_y)
        self.points_y = points_y
        
        points = np.zeros([points_num_x*points_num_y,2])
        for i in range(0, points_num_y):
            for j in range(0, points_num_x):
                points[i*points_num_x+j, 0] += [points_y[i]]
                points[i*points_num_x+j, 1] += [points_x[j]]
        self.points = points

        
        return(points)           
        
    
    
    def get_disp(self, points, method='lk'):
        """ Izpis koordinat v seznam.

            Funkcija vrne x in y koordinate.

            Argumenti:
            pot: pot do datoteke
        """
        self.video.set_points(points)
        self.video.set_method(method)
        disp = self.video.get_displacements(autosave=False)
        return(disp)
            
            
            
    def show_disp(self, disp, points, start_frame=1, scale=10.):
        """ Grafični prikaz premikov.

            Argumenti:
            disp: polje premikov
            start_frame: začetna sličica
            scale: velikost puščic
        """
        #field = disp[:,start_frame,:]
        width = 0.5
        fig, ax = plt.subplots(1)
        ax.imshow(self.mraw[start_frame], 'gray')
        ax.set_title('Prikaz pomikov na posnetku')
        max_L = np.max(disp[:,start_frame, 0]**2 + disp[:,start_frame, 1]**2)
        for i, ind in enumerate(points):
            f0 = disp[i,start_frame, 0]
            f1 = disp[i, start_frame, 1]
            alpha = (f0**2 + f1**2) / max_L
            if alpha < 0.2:
                alpha = 0.2
            plt.arrow(ind[1], ind[0], scale*f1, scale*f0, width=width, color='r', alpha=alpha)    
        

            
    def get_def(self, disp):  
        """ Izračuna specifične deformacije iz premikov.

            Funkcija vrne specifične deformacije.

            Argumenti:
            disp: polje pomikov 
         """          
        disp_x = disp_y = np.zeros([self.points_num_x, self.points_num_y, self.frame_num])
        disp_x = disp[:,:,1].reshape(self.points_num_y, self.points_num_x, self.frame_num)
        disp_y = disp[:,:,0].reshape(self.points_num_y, self.points_num_x, self.frame_num)
        
        def_x = np.zeros([self.points_num_y, self.points_num_x, self.frame_num])
        for i in range(0, self.frame_num):
            for j in range(0, self.points_num_y):
                for k in range(0, self.points_num_x-1):
                    def_x[j, k, i] += ((disp_x[j, k, i]-disp_x[j, k+1, i])/(2*self.ROI))
                                         
        def_y = np.zeros([self.points_num_y, self.points_num_x, self.frame_num])
        for i in range(0, self.frame_num):
            for j in range(0, self.points_num_y-1):
                for k in range(0, self.points_num_x):
                    def_y[j, k, i] += ((disp_y[j, k, i]-disp_y[j+1, k, i])/(2*self.ROI))
              
        sp_def = np.zeros([self.points_num_x*self.points_num_y,self.frame_num,2])
        for i in range(0, self.frame_num):
            for j in range(0, self.points_num_y):
                for k in range(0, self.points_num_x):
                    sp_def[j*self.points_num_x+k, i, 0] += def_y[j,k,i]
                    sp_def[j*self.points_num_x+k, i, 1] += def_x[j,k,i]
      
                        
        sp_def_y = sp_def[:,:,0].reshape(self.points_num_y, self.points_num_x, self.frame_num)

            
        sp_def_2d = np.zeros([self.points_num_x*self.points_num_y, self.frame_num,])
        for i in range(0, self.frame_num):
            sp_def_2d[:,i] += np.sqrt(sp_def[:,i,0]**2 + sp_def[:,i,1]**2)
        sp_def_2d = sp_def_2d.reshape(self.points_num_y, self.points_num_x, self.frame_num) 
        
        return(sp_def) 
    
        
    
    def refine_def(self, sp_def, axis=0, sum_num=5, freq=750, start_frame=None ,show_reference=True): #axis: 0=x, 1=y
        """ Izračuna izboljšanw specifične deformacije s seštevanjem zaporednih deformacij.

            Funkcija vrne izboljšane specifične deformacije.

            Argumenti:
            sp_def: polje specifičnih deformacij
            axis: os preračuna (x=0, y=1)
            sum_num: sumacijsko število - število seštetih sličic
            freq: frekvenca vzbujanja vzorca
            show_reference: izriše graf in prikaže kje sešteva sličice
         """   
        
        self.sum_num = sum_num
        if start_frame == None:
            reference_point = np.array([[(self.v_line1+self.v_line2)/2, (self.h_line1+self.h_line2)/2]])
            self.video.set_points(reference_point) #reference point is automaticly selected on center of aeria of interest
            self.video.set_method(method='lk')
            
            reference_disp = self.video.get_displacements(autosave=False) #izračun pomikov v referenčni točki
            x_disp = reference_disp[0,:,1]
            y_disp = reference_disp[0,:,0]
            
            max_frame = np.where(reference_disp[0,:,axis] == max(reference_disp[0,:,axis]))
            start_frame = int(max_frame[0])-int(self.frame_rate/(2*freq))
        else:
            start_frame = start_frame
        
        
        sp_def_2d = sp_def[:,:,axis].reshape(self.points_num_y, self.points_num_x, self.frame_num)
        ref_def = np.zeros([self.points_num_y, self.points_num_x])
        for i in range(start_frame, start_frame + sum_num):
            ref_def += sp_def_2d[:,:,i] 
            
        if show_reference == True:
            t = np.linspace(1, self.frame_num+1, self.frame_num, endpoint=False)
            fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
            ax1.imshow(self.mraw[start_frame].astype(float), 'gray')
            ax1.scatter(reference_point[:, 1], reference_point[:, 0], marker='x', color='tab:orange', label='Referenčn točka')
            ax1.set_title('Referenčna točka')
            ax2.plot(t, x_disp, label='Pomik x')
            ax2.plot(t, y_disp, label='Pomik y')
            ax2.axvline(x=start_frame, color='g', linestyle='--', label='začetek seštevanja')
            ax2.axvline(x=max_frame[0], color='r', linestyle='--', label='konec seštevanja')
            ax2.set_ylabel('Px')
            ax2.set_xlabel('sličice')
            ax2.set_title('Premik v času')
            plt.legend()
            plt.show()

        return(ref_def)
    
    
    
    def show_def(self,sp_def):
        """ Izris deformacij na contour grafu.

            Funkcija izriše in shrani graf specifičnih deformacij.

            Argumenti:
            sp_def: 2D polje specifičnih deformacij
        """
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(self.mraw[0].astype(float), 'gray')
        ax.contourf(self.points_x, self.points_y, abs(sp_def), alpha=.8)
        ax.set_title('Specifične deformacije')
        plt.show()
        name = f'{self.name}-ROI_{self.ROI}-sum_num_{self.sum_num}'
        plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
        
        
    def animacija(self, sp_def, axis=0, sum_num=3):
        
        frames = self.frame_num//sum_num
        sp_def_2d = sp_def[:,:,axis].reshape(self.points_num_y, self.points_num_x, self.frame_num)
        ani_def = np.zeros([self.points_num_y, self.points_num_x, frames])
        for i in range(0, frames-1):
            for j in range(0, sum_num):
                ani_def[:,:,i] += sp_def_2d[:,:,(sum_num*i)+j]
        
        self.ani_def = ani_def
        
        
        return(ani_def,self.points_x, self.points_y, self.mraw[0])

        


if __name__=='__main__':
#   
    name = 'NN3_desna_random_3_75'
    cih_file = f'/Users/nejc_novak/My Drive/Faks/Zaključna naloga/zaključna_naloga/NN_short/{name}.cihx'
    crack = Crack(cih_file, name)
    
    #points
    # points = crack.automatic_points(ROI_size=10)
    ROI = 10
    points = crack.rct_points(h_line1=170, h_line2=255, v_line1=55, v_line2=433, ROI_size=ROI)
    # crack.show_points(points)
    
    #pomiki
    disp = crack.get_disp(points)
    # crack.show_disp(disp, points, start_frame=5, scale=20.)
    
    #deformacije
    sp_def = crack.get_def(disp)
    
    
    #izbojšane def
    # SUM = 1
    # ref_def = crack.refine_def(sp_def, axis=0 ,sum_num=SUM, freq=750, show_reference=False)

    #prikaz deformacij
    # crack.show_def(ref_def)


    #animacija
    SUM_ani = 3
    ani_def, points_x, points_y, mraw_1 = crack.animacija(sp_def, axis=0, sum_num=SUM_ani)



    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1,figsize=(9,16))
    def animate(i):
            ax1.clear()
            ax1.imshow(mraw_1.astype(float), 'gray')
            ax1.contourf(points_x, points_y, abs(ani_def[:,:,i]), alpha=.8)
            ax1.set_title('Seštevek deformacij %03d'%(i)) 
            ax2.clear()
            ax2.plot(disp[1,:,1], color='orange')
            ax2.axvline(x=i*SUM_ani, linestyle='--', label='trenuta sličica')
            ax2.set_ylabel('Px')
            ax2.set_xlabel('sličice')
            ax2.set_title('Premik v času')

    interval = 0.3#in seconds     
    ani = animation.FuncAnimation(fig,animate,len(ani_def[0,0,:]),interval=interval*1e+3,blit=False)
    # ani.save('{name}_animation.mp4', writer = FFwriter)
    name1 = f'{name}-ROI_{ROI}-sum_num_{SUM_ani}_ani'
    ani.save(name1+'.gif', writer='imagemagick', fps=5)
    
    plt.show()
    






