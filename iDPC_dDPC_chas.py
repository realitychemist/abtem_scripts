# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:07:54 2022

@author: scaldero
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tifffile
import json
from PIL import Image
from tkinter import filedialog as fd
#import stemtool as st
#import py4DSTEM
import os

#%% OPEN emd

#file = QFileDialog.getOpenFileName()[0]
file= fd.askopenfilename()
#myList = os.listdir('<folder location>')

#%% Open file

f = h5py.File(file, 'r')

operations=list(f['Operations/StemInputOperation/'].items())
o=[]
detectorop=[]
images = []
detector = []
meta=[]
for i in range(len(operations)):
    o.append(f['Operations/StemInputOperation/'][operations[i][0]])
    metaData = o[i][0]
    detectorop.append(json.loads(metaData.decode('utf-8', 'ignore')))
    arr=np.zeros(f[detectorop[i]['dataPath']+'/Data'].shape, dtype='float32')
    detector.append(detectorop[i]['detector'])
    f[detectorop[i]['dataPath']+'/Data'].read_direct(arr)
    images.append(arr)
    tempMetaData = f[detectorop[i]['dataPath']+'/Metadata'][:, 0]
    validMetaDataIndex = np.where(tempMetaData > 0)
    metaData = tempMetaData[validMetaDataIndex].tobytes()
    meta.append(json.loads(metaData.decode('utf-8', 'ignore')))
    


#%% Selecting AC, BD and HAADF images

AC=images[detector.index('A-C')]
BD=images[detector.index('B-D')]
HAADF=images[detector.index('HAADF')]

#%%
def image_norm(image):
    '''Normalize intensity of a 1 channel image to the range 0-1.'''
    [min_, max_] = [np.min(image), np.max(image)]
    image = (image - min_)/(max_ - min_)
    return image



#%% Normalizing images

if len(AC.shape)>2:
    AC=image_norm(np.rollaxis(AC,2,0))
    BD=image_norm(np.rollaxis(BD,2,0))
    HAADF = image_norm(np.rollaxis(HAADF,2,0))

else:
    AC=image_norm(AC)
    #AC=(s[3].data/(s[3].data)-s[1].data)
    BD=image_norm(BD)
    #BD=(s[0].data-s[2].data)
    
#%% Only select a section of the stack

#AC=AC[10:19]
#BD=BD[10:19]
#HAADF=HAADF[10:19]


#%% Define the rotation angle (scanning angle and rotation of the deterctor)

rot_angle = float(meta[detector.index('A-C')]['CustomProperties']['DetectorRotation']['value']) #This is the detector rotation + scanrotation 
scanRotation = float(meta[detector.index('A-C')]['Scan']['ScanRotation'])# This is the scan roration used in Velox
rot_angle+=np.pi

#%%
pixel_size_x =meta[detector.index('A-C')]['BinaryResult']['PixelSize']['width']
pixel_size_y =meta[detector.index('A-C')]['BinaryResult']['PixelSize']['height']


#%% Save images (AC, BD and HAADF) as tiff

tifffile.imwrite(file.split('.emd')[0]+'_AC.tif', AC,metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])
tifffile.imwrite(file.split('.emd')[0]+'_BD.tif', BD,metadata=meta[detector.index('B-D')]['BinaryResult']['PixelSize'])
tifffile.imwrite(file.split('.emd')[0]+'_HAADF.tif', HAADF,metadata=meta[detector.index('HAADF')]['BinaryResult']['PixelSize'])

# #%%

# #%% Register the images. This step can be skipped if you want to calculate a full stack of images. 
# ##This uses STEMTOOLS developed by ORNL

# HAADF_obj = st.afit.multi_image_drift(HAADF)
# HAADF_obj.get_shape_stack()
# HAADF_corrected = HAADF_obj.corrected_stack()

# AC_obj=HAADF_obj
# AC_obj.image_stack=AC
# AC_obj.get_shape_stack()
# AC_corrected = AC_obj.corrected_stack()

# BD_obj=HAADF_obj
# BD_obj.image_stack=BD
# BD_obj.get_shape_stack()
# BD_corrected = BD_obj.corrected_stack()

# # #%%

# # AC=AC_corrected
# # BD=BD_corrected 
# # HAADF=HAADF_corrected 

# #%%

# tifffile.imwrite(file.split('.emd')[0]+'_AC_register.tif', AC,metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])
# tifffile.imwrite(file.split('.emd')[0]+'_BD_register.tif', BD,metadata=meta[detector.index('B-D')]['BinaryResult']['PixelSize'])
# tifffile.imwrite(file.split('.emd')[0]+'_HAADF_register.tif', HAADF,metadata=meta[detector.index('HAADF')]['BinaryResult']['PixelSize'])


#%%
'''Transform AC & BD to DPCx and DPCy'''
DPCx = (AC * np.cos(rot_angle) 
        - BD * np.sin(rot_angle))
    
DPCy = (AC * np.sin(rot_angle) 
        + BD * np.cos(rot_angle))

#%% Save DPC components
tifffile.imwrite(file.split('.emd')[0]+'_DPCx.tif', DPCx,metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])
tifffile.imwrite(file.split('.emd')[0]+'_DPCy.tif', DPCy,metadata=meta[detector.index('B-D')]['BinaryResult']['PixelSize'])



#%% SKIP THIS SECTION IF YOU DO NOT WANT TO ADD NOISE

# def add_noise(image, pixel_size=0.05, dose=1e9):
#     array = image_norm(image)
#     pixel_area=pixel_size**2
#     electrons_per_pixel = dose * pixel_area
#     noisy_data=np.random.poisson(array * electrons_per_pixel)
#     #tifffile.imwrite(file.split(".mrc")[0]+str(dose[i])+'.tif', np.array(noisy_data[i]))
#         #plt.imshow(noisy_data[0][0])
    
#     #tifffile.imwrite(file.split(".mrc")[0]+'.tif', image_norm(np.array(noisy_data)))
#     return noisy_data

# dose=1e3
# DPCx=add_noise(DPCx, dose=dose)
# DPCy=add_noise(DPCy, dose=dose)

# file=file.split('.emd')[0]+'_'+str(dose)+'.emd'

# %% Define the filter to apply using the first image 

 

#%% Calculate iDPC image and save it
sampling = 1
filt = 1e-3

f_freq_1d_y = np.fft.fftfreq(DPCx.shape[-2], sampling)
f_freq_1d_x = np.fft.fftfreq(DPCx.shape[-1], sampling)
k = np.meshgrid(f_freq_1d_x, f_freq_1d_y)
freqs = np.hypot(k[0], k[1])
angle = np.arctan2(k[1],k[0])
radi = np.where(freqs > filt , freqs, 0 )
k_sq = np.where(radi>0,(radi ** 2),1)
iDPCx = np.real(np.fft.ifft2((np.fft.fft2(DPCx) * radi * np.cos(angle)) 
                             / (2*np.pi * 1j * k_sq)))
iDPCy = np.real(np.fft.ifft2((np.fft.fft2(DPCy) * radi * np.sin(angle)) 
                             / (2*np.pi * 1j * k_sq)))
iDPC= iDPCx + iDPCy
iDPC = iDPC.astype('float32')

tifffile.imwrite(file.split('.emd')[0]+'_iDPC.tif', iDPC,metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])

#%% Calculate iDPC using py4DSTEM

#phase = py4DSTEM.process.dpc.dpc.get_phase_from_CoM(AC, BD, rot_angle, False, regLowPass=0.5, regHighPass=100, paddingfactor=2, stepsize=1, n_iter=10, phase_init=None)

#tifffile.imwrite(file.split('.emd')[0]+'_phase.tif', iDPC,metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])


#%% Calculate the dDPC image and save it
gradx=np.gradient(DPCx, axis=-1)
#grady=np.gradient(np.flip(eDPCy,axis=-1), axis=-2) # the image need to be flipped because the gradient is calculated in the opposit direction
grady=np.gradient(DPCy, axis=-2) 
dDPC=-(gradx+grady)
dDPC = dDPC.astype('float32')

tifffile.imwrite(file.split('.emd')[0]+'_dDPC.tif', dDPC, metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])

#%% Calculate an image proportional to Electric field Mag and direction without right scale

DPCx-=np.min(DPCx)+(np.max(DPCx)-np.min(DPCx))/2
DPCy-=np.min(DPCy)+(np.max(DPCy)-np.min(DPCy))/2

eDPCx=-1*DPCx
eDPCy=-1*DPCy
eDPC=np.sqrt(eDPCx**2+eDPCy**2) # I need to multiply by the factors
eDPCan=np.arctan2(eDPCy, eDPCx)

eDPCa=255*(eDPCan-np.min(eDPCan))/(np.max(eDPCan)-np.min(eDPCan))
sat=np.ones(eDPC.shape)*255
eDPC=255*(eDPC-np.min(eDPC))/(np.max(eDPC)-np.min(eDPC))

eDPC=np.flip(np.array(eDPC,dtype="uint8"),axis=-2)
eDPCa=np.flip(np.array(eDPCa,dtype="uint8"),-2)
sat=np.flip(np.array(sat,dtype="uint8"),-2)

tifffile.imwrite(file.split('.emd')[0]+'_eDPC.tif', eDPC, metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])

#  Direction

h=[]
s=[]
v=[]
efield=[]

if len(eDPC.shape)>=3:
    for x in range(eDPC.shape[0]):
        h.append(Image.fromarray(eDPCa[x], mode=None)) # mode I
        s.append(Image.fromarray(sat[x], mode=None))
        v.append(Image.fromarray(eDPC[x], mode=None))
        efield.append(Image.merge("HSV",(h[x],s[x],v[x])))
        efield[x]=np.array(efield[x].convert(mode="RGB"))
else:
    h.append(Image.fromarray(eDPCa, mode=None)) # mode I
    s.append(Image.fromarray(sat, mode=None))
    v.append(Image.fromarray(eDPC, mode=None))
    efield.append(Image.merge("HSV",(h[0],s[0],v[0])))
    efield[0]=np.array(efield[0].convert(mode="RGB"))

tifffile.imwrite(file.split('.emd')[0]+'_eDPC_dir.tif', efield, metadata=meta[detector.index('A-C')]['BinaryResult']['PixelSize'])
#efield=Image.merge("HSV",(h,s,v))

