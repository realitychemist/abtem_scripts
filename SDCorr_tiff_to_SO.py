# %% SETUP
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

import SingleOrigin as so

from scanning_drift_corr import SPmerge01linear as SP01
from scanning_drift_corr import SPmerge02 as SP02
from scanning_drift_corr import SPmerge03 as SP03
from quickcrop import gui_crop

plt.rcParams['figure.figsize'] = [8.0, 8.0]
plt.rcParams['figure.dpi'] = 300

# %% IMPORT IMAGE FILES

BASE_PATH = r"E:\Users\Charles\20220701_AutoExport"
fnames = ["20220701 1816 15.5 Mx HAADF 6.55 nm HAADF.tif",
          "20220701 1816 15.5 Mx HAADF 6.55 nm 0001 HAADF.tif",
          "20220701 1816 15.5 Mx HAADF 6.55 nm 0002 HAADF.tif",
          "20220701 1816 15.5 Mx HAADF 6.55 nm 0003 HAADF.tif"]
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

scanAngles = (0, 90, 180, 270)  # Same order as input images
smerge = SP01.SPmerge01linear(scanAngles, *images)
print("Initial correction done!")

# %% NONLINEAR REFINEMENT

smerge = SP02.SPmerge02(smerge, 5, 8)
print("Nonlinear refinement done!")

# %% FINAL MERGE

# Can take a couple minutes, has no progress bar
image_corrected, signal_array, density_array = SP03.SPmerge03(smerge)
print("Final merge done!")

# %% BIT DEPTH CORRECTION

# Convert format for export to 16-bit tiff
image_final = ((image_corrected - np.min(image_corrected)) /
               (np.max(image_corrected) - np.min(image_corrected))) * 65535
image_final = image_final.astype(np.uint16)

# %% CROP SQUARE

image_cropped = gui_crop(image_final)

# %% EXPORT INTERMEDIATE

# Run this cell if you want to save the image before column finding
EXPORT_NAME = "20220701 1816 15.5 Mx HAADF 6.55 nm HAADF SDCorr.tif"
tif.imwrite(os.path.join(BASE_PATH, "processed", EXPORT_NAME), data=image_cropped)

# %% LOAD INTERMEDIATE

# Run this cell if you want to load an already corrected and cropped image for analysis
IMPORT_PATH = r"E:\Users\Charles\20220524_AutoExport\processed"
IMPORT_NAME = "20220524 5.40 Mx HAADF SDCorr.tif"
image_cropped = np.array(tif.imread(os.path.join(IMPORT_PATH, IMPORT_NAME)))

# %% SINGLEORIGIN INITIALIZATION

CIF_PATH = r"E:\Users\Charles\BaTiO3 Controls\No Shift\BaTiO3_99_noshift.cif"

za = [1, 0, 0]  # Zone axis direction
a2 = [0, 1, 0]  # Apparent horizontal axis in projection
a3 = [0, 0, 1]  # Most vertical axis in projection

# Initialize UnitCell object
uc = so.UnitCell(CIF_PATH)
uc.transform_basis(za, a2, a3)

# Project unit cell, combine coincident columns
# Ignore light elements for HAADF
uc.project_uc_2d(proj_axis=0, ignore_elements=['O'])
uc.combine_prox_cols(toler=1e-2)
uc.plot_unit_cell()  # Check this output to make sure it's sensible

# %% BASIS VECTOR DEFINITION + REF LATTICE FIT

# Initialize AtomicColumnLattice object
acl = so.AtomicColumnLattice(image_cropped, uc, resolution=0.8,
                             xlim=None, ylim=None)

# Get real space basis vectors using the FFT
# If some FFT peaks are weak or absent (such as forbidden reflections),
#  specify the order of the first peak that is clearly visible
acl.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)
# Pick out an atom from the supplied image for an initial fit
acl.define_reference_lattice()

# %% ATOM COLUMN FITTING

# Fit atom columns at reference lattice points
acl.fit_atom_columns(buffer=10, local_thresh_factor=0.95,
                     grouping_filter=None, diff_filter="auto")
# Check results (including residuals) to verify accuracy!
print("Atom column fitting done!")

# %% BASIS VECTOR FITTING

# Use the fitted atomic column positions to refine the basis vectors and origin.  It is best to
#  choose a sublattice with minimal displacements.  It also must have only one column per
#  projected unit cell.  If no sublattice meets this criteria, specify a specific column in the
#  projected cell.
acl.refine_reference_lattice(filter_by='elem', sites_to_use='Ba', outliers=30)
print("Basis vector fitting done!")

# %% CHECK RESIDUALS

# Ideal == small and relatively flat
acl.plot_fitting_residuals()

# %% PLOT COLUMN POSITIONS

acl.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                               fit_or_ref='fit', outliers=30,
                               plot_masked_image=True)

# %% PLOT DISPLACEMENTS FROM REF LATTICE

acl.plot_disp_vects(filter_by='elem', sites_to_plot='all', titles=None,
                    x_crop=[0, acl.w], y_crop=[acl.h, 0],
                    # x_crop=[0, 500], y_crop=[500, 0],
                    scalebar=True, scalebar_len_nm=2,
                    max_colorwheel_range_pm=None,
                    plot_fit_points=False, plot_ref_points=False)

# %% COSMETIC: IMAGE & DATA ROTATION

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')

acl_rot.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                                   fit_or_ref='fit',
                                   plot_masked_image=False)

acl_rot.plot_disp_vects(filter_by='elem', sites_to_plot='all', titles=None,
                        x_crop=[0, acl_rot.w], y_crop=[acl_rot.h, 0],
                        scalebar=True, scalebar_len_nm=2,
                        max_colorwheel_range_pm=25)
