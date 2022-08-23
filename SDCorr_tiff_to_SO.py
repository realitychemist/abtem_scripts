# %% SETUP
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

import SingleOrigin as so

from scanning_drift_corr import SPmerge01linear as SP01
from scanning_drift_corr import SPmerge02 as SP02
from scanning_drift_corr import SPmerge03 as SP03

from scipy import spatial
from matplotlib import cm
from matplotlib.colors import Normalize
from quickcrop import gui_crop

from pysal.explore import esda
from pysal.lib import weights

plt.rcParams['figure.figsize'] = [8.0, 8.0]
plt.rcParams['figure.dpi'] = 300

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

image_cropped = gui_crop(image_final)

# %% EXPORT INTERMEDIATE

# Run this cell if you want to save the image before column finding
EXPORT_NAME = "20220524 5.40 Mx HAADF SDCorr.tif"
tif.imwrite(os.path.join(BASE_PATH, "processed", EXPORT_NAME), data=image_cropped)

# %% LOAD INTERMEDIATE

# Run this cell if you want to load an already corrected and cropped image for analysis
IMPORT_PATH = r"E:\Users\Charles\20220524_AutoExport\processed"
IMPORT_NAME = "20220524 5.40 Mx HAADF SDCorr.tif"
image_cropped = np.array(tif.imread(os.path.join(IMPORT_PATH, IMPORT_NAME)))

# %% SINGLEORIGIN INITIALIZATION

CIF_PATH = r"E:\Users\Charles\BaTiO3 Controls\No Shift\BaTiO3_99_noshift.cif"

za = [0, 0, 1]  # Zone axis direction
a2 = [1, 0, 0]  # Apparent horizontal axis in projection
a3 = [0, 1, 0]  # Most vertical axis in projection

# Initialize UnitCell object
uc = so.UnitCell(CIF_PATH)
uc.transform_basis(za, a2, a3)

# Project unit cell, combine coincident columns
# Ignore light elements for HAADF
uc.project_uc_2d(proj_axis=0, ignore_elements=["O"])
uc.combine_prox_cols(toler=1e-2)
uc.plot_unit_cell()  # Check this output to make sure it's sensible

# %% BASIS VECTOR DEFINITION + REF LATTICE FIT

# Initialize AtomicColumnLattice object
image_cropped = so.image_norm(image_cropped)
acl = so.AtomicColumnLattice(image_cropped, uc, probe_fwhm=0.8,
                             origin_atom_column=0)

# Get real space basis vectors using the FFT
# If some FFT peaks are weak or absent (such as forbidden reflections),
#  specify the order of the first peak that is clearly visible
acl.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)
# Pick out an atom from the supplied image for an initial fit
acl.define_reference_lattice()

# %% ATOM COLUMN FITTING

# Fit atom columns at reference lattice points
acl.fit_atom_columns(buffer=20, local_thresh_factor=0.95,
                     grouping_filter="auto", diff_filter="auto", parallelize=False)
# Check results (including residuals) to verify accuracy!
print("Atom column fitting done!")

# %% BASIS VECTOR FITTING

# Use the fitted atomic column positions to refine the basis vectors and origin.  It is best to
#  choose a sublattice with minimal displacements.  It also must have only one column per
#  projected unit cell.  If no sublattice meets this criteria, specify a specific column in the
#  projected cell.
acl.refine_reference_lattice(filter_by='elem', sites_to_use='Ba')
print("Basis vector fitting done!")

# %% CHECK RESIDUALS

# Ideal == small and relatively flat
acl.get_fitting_residuals()

# %% PLOT COLUMN POSITIONS

acl.plot_atom_column_positions(scatter_kwargs_dict={"s": 5}, scalebar_len_nm=None)

# %% OPTIONAL COSMETIC: IMAGE & DATA ROTATION

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')

acl_rot.plot_atom_column_positions(scatter_kwargs_dict={"s": 5}, scalebar_len_nm=None)

acl_rot.plot_disp_vects(scalebar=True, scalebar_len_nm=2,
                        max_colorwheel_range_pm=10)

# %% COMPUTE INTENSITY RATIOS

posint_df = acl.at_cols[["elem", "u", "v", "x_fit", "y_fit", "x_ref",
                         "y_ref", "total_col_int"]].reset_index(drop=True)


def _format_neighbors(row):
    """Helper function to reformat neighbor array to use the correct indices."""

    if row["elem"] == "Ba":
        return None

    _, idxs = row["neighbors"]
    # TODO: Automate finding the no-near-neighbor index
    idxs = list(filter(lambda x: x != 618, idxs))
    new_idxs = []
    for i in idxs:
        x, y = kdtree_a_sites.data[i]
        match = posint_df.loc[posint_df["x_fit"] == x]
        if len(match) != 1:
            raise RuntimeError("Something went wrong!")
        new_idxs.append(match.index[0])
    return new_idxs


def _get_int_ratios(row):
    """Helper function to get column intensity ratios (B/A)."""

    if row["neighbors"] is not None:
        neighboring_intensities = []
        for i in row["neighbors"]:
            neighboring_intensities.append(posint_df["total_col_int"].iloc[i])
        return row["total_col_int"] / np.mean(neighboring_intensities)


# For each Ti, find 4 nearest Ba...
# KDTree for fast kNN lookup
kdtree_a_sites = spatial.KDTree(posint_df.loc[posint_df["elem"] == "Ba", "x_fit":"y_fit"])

# Find the 4 nearest neighbors; A sites will have their neighbors set to None
# distance_upper_bound=55 is a hyperparameter, should avoid inclusion of A site columns from
#  distant unit cells when the B site column is near an edge
posint_df["neighbors"] = posint_df.apply(lambda row: kdtree_a_sites.query(
    (row["x_fit"], row["y_fit"]), k=4, distance_upper_bound=55), axis=1)
posint_df["neighbors"] = posint_df.apply(_format_neighbors, axis=1)
# Compute the intensity ratios
posint_df["int_ratio"] = posint_df.apply(_get_int_ratios, axis=1)
posint_df["disp"] = posint_df.apply(lambda r: (
    r["x_fit"]-r["x_ref"], r["y_fit"]-r["y_ref"]), axis=1)


# %% PLOT RELEVANT DATA

# For each Ti column, extract the coordinates and make a Voronoi object
bpts = posint_df.loc[posint_df["elem"] == "Ti",
                     ["u", "v", "x_fit", "y_fit"]].reset_index(drop=True)

# Create normalized colormap from ratios
ratios = posint_df.loc[posint_df["elem"] == "Ti", "int_ratio"].reset_index(drop=True)
rmax = max(ratios)
rmin = min(ratios)
norm = Normalize(vmin=rmin, vmax=rmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Spectral_r)

disps = posint_df.loc[posint_df["elem"] == "Ti", "disp"].reset_index(drop=True)

# Plot the Voronoi map over the image
# fig, ax = acl.plot_disp_vects(sites_to_plot="all")

fig, ax = plt.subplots()
ax.imshow(image_cropped, cmap="gray")

for i, b in bpts.iterrows():
    ax.scatter(b["x_fit"], b["y_fit"], s=12, color=mapper.to_rgba(
        ratios[i]), linewidths=0, alpha=1)

ax.axis("off")
fig.colorbar(mapper, label="B/A Intensity Ratio")

# %% PYSAL ANALYSIS


for idx, (_, site) in enumerate(bpts.iterrows()):
    neighbor_umn = bpts[(bpts.v == site.u - 1) & (bpts.u == site.v)]
    neighbor_upl = bpts[(bpts.v == site.u + 1) & (bpts.u == site.v)]
    neighbor_vmn = bpts[(bpts.v == site.u) & (bpts.v == site.v - 1)]
    neighbor_vpl = bpts[(bpts.v == site.u) & (bpts.v == site.v + 1)]
    for n in [neighbor_umn, neighbor_upl, neighbor_vmn, neighbor_vpl]:
        if n is not None:
            pass
