import os
import copy
import h5py
import PyQt5
import matplotlib.pyplot as plt
import numpy as np
# from scipy.ndimage import rotate
import json
from tifffile import imwrite
import re
import SingleOrigin as so
import scanning_drift_corr.api as sdc


print('Select source folder')
source_folder = PyQt5.QtWidgets.QFileDialog.getExistingDirectory()


# %% SOME INITIAL SETTINGS
get_dpc_images = False
num_frames_to_use = 2
grd = 'NOGRD'    # Default filename component if no scan calibration used

# %% Get all .h5 filenames in the directory
file_list = []
for file in os.listdir(source_folder):
    if file[-3:] == '.h5':
        file_list += [file]
    else:
        continue
file_list.sort()

# %% CHOOSE IMAGE FROM THE DIRECTORY LIST TO LOAD
filename = file_list[5]  # Change this to go through the files

f = h5py.File(os.path.join(source_folder, filename), 'r')
scanAngles = np.array(f['Frame Rotation'])
if num_frames_to_use <= scanAngles.shape[0]:
    scanAngles = scanAngles[:num_frames_to_use]
else:
    num_frames_to_use = scanAngles.shape[0]

# Load HAADF images
haadf = np.array(f['Images']['HAADF'][
    re.search(r"'(.*)'", repr(str(f['Images']['HAADF'].keys()))).group(1)])

# Rotate images for scanning_drift_corr
haadf = np.array([np.rot90(haadf[:, :, i], -(a//90)) for i, a in enumerate(scanAngles)])

# haadf = np.array([so.fast_rotate_90deg(haadf[:,:,i], -ang) for i, ang
#                     in enumerate(scanAngles)])

if get_dpc_images:
    if np.isin('Segment1 - Segment3', list(f['Images'].keys())):
        print('DPC images in file')
        dpc_found = True
        # Load DPC difference signals
        dpc_ac = np.array(f['Images']['Segment1 - Segment3'][
            re.search(r"'(.*)'", repr(str(f['Images']['Segment1 - Segment3'
                                                      ].keys()))).group(1)]
            )[:num_frames_to_use]
        dpc_bd = np.array(f['Images']['Segment2 - Segment4'][
            re.search(r"'(.*)'", repr(str(f['Images']['Segment2 - Segment4'].keys()))).group(1)]
            [:num_frames_to_use])
        # Rotate images for scanning_drift_corr
        dpc_ac = np.array([so.fast_rotate_90deg(dpc_ac[:, :, i], -ang) for i, ang
                           in enumerate(scanAngles)])
        dpc_bd = np.array([so.fast_rotate_90deg(dpc_bd[:, :, i], -ang) for i, ang
                           in enumerate(scanAngles)])

        dpc_ac = so.image_norm(dpc_ac)
        dpc_bd = so.image_norm(dpc_bd)
        dpc_mag = (dpc_ac**2 + dpc_bd**2)**0.5

else:
    dpc_found = False


metadata = json.loads(f['Metadata'].asstr()[0])

# Extract microscope parameters:
pixel_size = float(metadata['Pixel Size [m]'][0])/1e-9
conv_angle = float(metadata['Convergence Semi-angle [mrad]'])*1e-3

haadf = so.image_norm(haadf[:num_frames_to_use])

# %%  DISPLAY FIRST TWO IMAGES IN mit-stem SERIES
plt.figure()
# plt.imshow(dpc_mag[:,:,0])
plt.imshow(haadf[0])
plt.figure()
# plt.imshow(dpc_mag[:,:,0])
plt.imshow(haadf[1])
# %% RUN SPMERGE01LINEAR

# Note: Depending on the image, it may be better to correct the HAADF image
# and apply the correciton to the DPC component images. Alternatively, the dpc
# magnitude can be used. Pass the desired image to SPmerge01linear.


sm = sdc.SPmerge01linear(scanAngles, haadf, niter=1)

# %% LOAD CALIBRATION FILE AND CORRECT SCAN VECTOR DISTORTION

# path, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(
#     caption='Select an image to load...',
#     filter="numpy (*.npy)")
# print(f'path to imported image: {path}')
# m_calibration = np.load('/Users/funni/Documents/Academics/Grad_School/Research/Python Scripts/GRD Calibrations/m_calibration_Themis_CMU_20220907_3.6M_2k.npy')
# sm = SPcalibration(sm, m_calibration)
# grd = 'GRD'     # Added to filename since scan calibration completed
# print('Affine transformation applied. Ensure correct calibration was used.')

# %% RUN SPMERGE02

sm2 = sdc.SPmerge02(sm, 16, 8, flagGlobalShift=True, stepSizeReduce=0.5)

# %% RUN SPMERGE03: GET FINAL IMAGE

imageFinal, _, _ = sdc.SPmerge03(sm2, KDEsigma=0.5)

# %% APPLY CORRECTION TO DPC COMPONENTS

# sigma = 0.5
# sm_ac = copy.deepcopy(sm)
# sm_ac.scanLines = dpc_ac
# imageFinal_ac, _, _ = sdc.SPmerge03(sm_ac, KDEsigma=sigma)

# sm_bd = copy.deepcopy(sm)
# sm_bd.scanLines = dpc_bd
# imageFinal_bd, _, _ = sdc.SPmerge03(sm_bd, KDEsigma=sigma)


# %% CHECK CROPPING

haadf_drift_corr = so.image_norm(imageFinal)
plt.figure()
# plt.imshow(dpc_mag[:,:,0])
plt.imshow(haadf_drift_corr)
# %% CROP DPC IMAGES AND GET PHASE IMAGE

# CoMxy = np.stack([imageFinal_ac, imageFinal_bd])[:, 220:-170, 170:-170]

# phase = -so.get_phase_from_com(CoMxy, theta=-132, flip=True,
#                                high_low_filter=True,
#                                filter_params={'beam_energy': 200e3,
#                                               'conv_semi_angle': conv_angle,
#                                               'pixel_size': pixel_size,
#                                               'high_pass': 0.05,
#                                               'low_pass': 0.85,
#                                               'edge_smoothing': 0.02})
# phase = so.image_norm(phase)
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.imshow(phase)
# %% SAVE IMAGES
# if dpc_found:
#     imwrite(os.path.join(source_folder,
#                          filename[:-3] + f'drift_corr_DPC_phase_{grd}' + '.tif'),
#             (phase*(2**16-1)).astype(np.uint16), photometric='minisblack')


imwrite(os.path.join(source_folder,
                     filename[:-3] + f'drift_corr_HAADF_{grd}' + '.tif'),
        (haadf_drift_corr*(2**16-1)).astype(np.uint16),
        photometric='minisblack')
