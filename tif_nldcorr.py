# IMPORTS
import os
import numpy as np
import tifffile as tif

from scanning_drift_corr import SPmerge01linear as SP01
from scanning_drift_corr import SPmerge02 as SP02
from scanning_drift_corr import SPmerge03 as SP03

import quickcrop.quickcrop as qc

# %% IMPORT IMAGE FILES

BASE_PATH = r"E:\Users\Charles\20220524_AutoExport"
fnames = ["1557 20220524 DPC 7.60 Mx HAADF-DF4 HAADF.tif",
          "1558 20220524 DPC 7.60 Mx HAADF-DF4 HAADF.tif",
          "1558 20220524 DPC 7.60 Mx HAADF-DF4 0001 HAADF.tif",
          "1559 20220524 DPC 7.60 Mx HAADF-DF4 HAADF.tif"]
images = [np.array(tif.imread(os.path.join(BASE_PATH, fname))) for fname in fnames]
imsize = [np.shape(image) for image in images]
# Sanity check: all images must be the same resolution and must be square
if not all((len(set(size)) == 1 for size in imsize)):  # Squareness check
    import sys
    sys.tracebacklimit = 0
    raise RuntimeError("At least one imported image is non-square; "
                       "all images should be square and the same size.  "
                       f"Sizes: {imsize}")
if not len(set(imsize)) == 1:
    import sys
    sys.tracebacklimit = 0
    raise RuntimeError("At least one imported image has a size differing from the others; "
                       "all images should be square and the same size.  "
                       f"Sizes: {imsize}")

# %% INITIAL LINEAR CORRECTION

scanAngles = (0, -90, -180, -270)  # Same order as input images
smerge = SP01.SPmerge01linear(scanAngles, *images)
print("Initial correction done!")

# %% NONLINEAR REFINEMENT

smerge = SP02.SPmerge02(smerge, 8, 8)
print("Nonlinear refinement done!")

# %% FINAL MERGE

# Can take a couple minutes, has no progress bar
image_corrected, signal_array, density_array = SP03.SPmerge03(smerge, KDEsigma=0.5)
print("Final merge done!")

# %% BIT DEPTH CORRECTION

# Convert format for export to 16-bit tiff
image_final = ((image_corrected - np.min(image_corrected)) /
               (np.max(image_corrected) - np.min(image_corrected))) * 65535
image_final = image_final.astype(np.uint16)

# %% CROP SQUARE

image_cropped = qc.gui_crop(image_final)

# %% EXPORT

# Run this cell if you want to save the image before column finding
EXPORT_NAME = "20220524 5.40 Mx HAADF SDCorr.tif"
tif.imwrite(os.path.join(BASE_PATH, "processed", EXPORT_NAME), data=image_cropped)
