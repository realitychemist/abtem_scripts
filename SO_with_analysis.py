"""
1. Read in a STEM image
2. Create a model of the structure in the image
3. Fit the atom columns using SingleOrigin
4. Perform statistical analysis!
"""

# %% IMPORTS
import os
import numpy as np
import tifffile as tif
import SingleOrigin.SingleOrigin as so
from copy import deepcopy


# %% LOAD DRIFT CORRECTED IMAGE

IMPORT_PATH = r"C:\Users\charles\Documents\AlScN\img"
IMPORT_NAME = "AlScN0.5.tif"
image_cropped = np.array(tif.imread(os.path.join(IMPORT_PATH, IMPORT_NAME)))
# image_cropped = qc.gui_crop(image_cropped)
# image_cropped = so.image_norm(image_cropped)

# _, image_cropped = divide_image_frequencies(image_cropped, s=350, show_images=True)
image_cropped = so.image_norm(image_cropped)

# %% SINGLEORIGIN INITIALIZATION

CIF_PATH = r"C:\Users\charles\Documents\AlScN\raw\AlN.cif"

za = [1, 1, 0]  # Zone axis direction
a1 = [-1, 1, 0]  # Apparent horizontal axis in projection
a2 = [0, 0, -1]  # Most vertical axis in projection

uc = so.UnitCell(CIF_PATH, origin_shift=[0, 0, 0])
uc.atoms.replace("Ti/O", "Ti/Zr", inplace=True)

# Ignore light elements for HAADF
uc.project_zone_axis(za, a1, a2, ignore_elements=["N"])
uc.combine_prox_cols(toler=1e-2)
# uc.plot_unit_cell()  # Uncomment and check this output if changing the u.c.

hr_img = so.HRImage(image_cropped)
lattice = hr_img.add_lattice("BZT", uc)

# Tidy namespace
# del za, a1, a2, uc, IMPORT_PATH, IMPORT_NAME, CIF_PATH

# %% PREPARE FOR FITTING
# NOTE: There are a couple of steps here that require interaction and will time out if ignored

# If some FFT peaks are weak or absent (such as forbidden reflections),
#  specify the order of the first peak that is clearly visible
lattice.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)

# lattice.get_roi_mask_std(r=15, buffer=20, thresh=0.25, show_mask=True)
lattice.roi_mask = np.ones(image_cropped.shape)  # For pre-cropped images

lattice.define_reference_lattice()

# %% ATOM COLUMN FITTING

lattice.fit_atom_columns(buffer=0, local_thresh_factor=0, use_background_param=True,
                         use_bounds=True, use_circ_gauss=False, parallelize=True,
                         peak_grouping_filter=None)

# Must have only one column per projected unit cell.  If no sublattice meets this criteria,
#  specify a specific column in the projected cell.
lattice.refine_reference_lattice(filter_by='elem', sites_to_use='Ba')

# %% CHECK RESIDUALS

# Ideal == small and relatively flat
# This can be skipped if you already know you've fit the image well
lattice.get_fitting_residuals()

# %% PLOT COLUMN POSITIONS

hr_img.plot_atom_column_positions(scatter_kwargs_dict={"s": 20}, scalebar_len_nm=None,
                                  color_dict={"Ba": "#FF0060", "Ti/Zr": "#84BABA"},
                                  outlier_disp_cutoff=100, fit_or_ref="ref")

# %% PLOT DISPLACEMNT VECTORS

hr_img.plot_disp_vects(sites_to_plot=["Ti/Zr"], arrow_scale_factor=2)

# %% CREATE FRAME

# A copy of the at_cols dataframe of the lattice object, for convenience
frame = deepcopy(lattice.at_cols)

frame.drop(["site_frac", "x", "y", "weight"], axis=1, inplace=True)  # We don't need these cols
frame.reset_index(drop=True, inplace=True)

reject_outliers(frame, mode="total_col_int")
normalize_intensity(frame, lattice.a_2d, n=8, method="global_minmax", kind="Ti/Zr")
disp_calc(frame, normalize=True, kind="Ti/Zr")
drop_elements(frame, ["Ba"])  # Doing this messes up the indexing in `neighborhood`
# So we'll rebuild the neighborhood in the new indexing
tree = grow_tree(frame, lattice.a_2d, "Ti/Zr")
frame["neighborhood"] = frame.apply(lambda row: get_near_neighbors(row, frame, lattice.a_2d,
                                                                   tree, n=8), axis=1)

# %% PYSAL ANALYSIS
sts = get_stats(frame, "queen", lattice.a_2d, kind="geary_local",
                columns=["dot_normal_disp"], p=10000, printstats=True)
add_stats_to_frame(frame, sts, "geary")

# %% PLOT GEARY
plot_geary_clusters(frame, image_cropped, sig=0.05)
