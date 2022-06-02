from scanning_drift_corr import SPmerge01linear as SP01
from scanning_drift_corr import SPmerge02 as SP02
from scanning_drift_corr import SPmerge03 as SP03

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import os

plt.rcParams['figure.figsize'] = [8.0, 8.0]
plt.rcParams['figure.dpi'] = 300

# %% IMPORT IMAGE FILES
if __name__ == "__main__":
    base_path = r"E:\Users\Charles\20220524_AutoExport"
    fnames = ["1547 20220524 DPC 5.40 Mx HAADF-DF4 HAADF.tif",
              "1548 20220524 DPC 5.40 Mx HAADF-DF4 HAADF.tif",
              "1548 20220524 DPC 5.40 Mx HAADF-DF4 0001 HAADF.tif",
              "1549 20220524 DPC 5.40 Mx HAADF-DF4 HAADF.tif"]
    images = [np.array(tif.imread(os.path.join(base_path, fname))) for fname in fnames]

    # %% INITIAL LINEAR CORRECTION
    scanAngles = (0, 90, 180, 270)
    smerge = SP01.SPmerge01linear(scanAngles, *images)

    # %% NONLINEAR REFINEMENT
    smerge = SP02.SPmerge02(smerge, 12, 8)

    # %% FINAL MERGE
    image_final, signal_array, density_array = SP03.SPmerge03(smerge)

    # %% EXPORT
    export_name = "20220524 5.40 Mx HAADF SDCorr_imshift1.tif"

    # Convert format for export to 16-bit tiff
    image_final_export = ((image_final - np.min(image_final)) /
                          (np.max(image_final) - np.min(image_final))) * 65535
    image_final_export = image_final_export.astype(np.uint16)

    tif.imwrite(os.path.join(base_path, export_name), data=image_final_export)
