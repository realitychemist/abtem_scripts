# %% SETUP
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

import SingleOrigin as so

from scipy import spatial, stats, optimize
from copy import deepcopy
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgba

from pysal.explore import esda
from pysal.lib import weights

# %% LOAD DRIFT CORRECTED IMAGE

IMPORT_PATH = r"E:\Users\Charles\bzt_data_100_20221212"
IMPORT_NAME = "12-12-2022_17.41.15_HAADFdrift_corr_HAADF_NOGRD.tif"
image_cropped = np.array(tif.imread(os.path.join(IMPORT_PATH, IMPORT_NAME)))
# image_cropped = gui_crop(image_cropped)  # If a mask method other than STD needs to be used
image_cropped = so.image_norm(image_cropped)

# %% SINGLEORIGIN INITIALIZATION

CIF_PATH = r"E:\Users\Charles\BZT.cif"

za = [1, 0, 0]  # Zone axis direction
a1 = [0, 1, 0]  # Apparent horizontal axis in projection
a2 = [0, 0, 1]  # Most vertical axis in projection

# Initialize UnitCell object
uc = so.UnitCell(CIF_PATH, origin_shift=[0, 0, 0])
uc.atoms.replace(["Sr", "Ti"], ["Ba", "Ti/Zr"], inplace=True)

# Project unit cell, combine coincident columns
# Ignore light elements for HAADF
uc.project_zone_axis(za, a1, a2, ignore_elements=["O"])
uc.combine_prox_cols(toler=1e-2)
uc.plot_unit_cell()  # Check this output to make sure it's sensible

# %% CREATE HRIMAGE OBJECT
hr_img = so.HRImage(image_cropped)
lattice = hr_img.add_lattice("BZT", uc)

# %% BASIS VECTOR DEFINITION

# Get real space basis vectors using the FFT
# If some FFT peaks are weak or absent (such as forbidden reflections),
#  specify the order of the first peak that is clearly visible
lattice.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)

# %% DEFINE REGION MASK
lattice.get_region_mask_std(r=15, buffer=15, thresh=0.25)
# lattice.region_mask = np.ones(image_cropped.shape)

# %% REFERENCE LATTICE FIT
lattice.define_reference_lattice()

# %% ATOM COLUMN FITTING

# Fit atom columns at reference lattice points
lattice.fit_atom_columns(buffer=10, local_thresh_factor=0.1,
                         use_circ_gauss=False, parallelize=True)
# Check results (including residuals) to verify accuracy!
print("Atom column fitting done!")

# %% BASIS VECTOR FITTING

# Use the fitted atomic column positions to refine the basis vectors and origin.  It is best to
#  choose a sublattice with minimal displacements.  It also must have only one column per
#  projected unit cell.  If no sublattice meets this criteria, specify a specific column in the
#  projected cell.
lattice.refine_reference_lattice(filter_by='elem', sites_to_use='Ba')
print("Basis vector fitting done!")

# %% CHECK RESIDUALS

# Ideal == small and relatively flat
lattice.get_fitting_residuals()

# %% PLOT COLUMN POSITIONS

hr_img.plot_atom_column_positions(scatter_kwargs_dict={"s": 20}, scalebar_len_nm=None,
                                  color_dict={"Ba": "#FF0060", "Ti/Zr": "#84BABA"},
                                  outlier_disp_cutoff=100, fit_or_ref="ref")

# hr_img.plot_disp_vects()

# %% CREATE BFRAME

# A copy of the at_cols dataframe of the lattice object, for convenience
bframe = deepcopy(lattice.at_cols)

# %% AUGMENT DATAFRAME


def _fmt_neighbors(row, df, kdtree):
    """Helper function to reformat neighbor array to use the correct indices."""
    _, idxs = row["neighbors"]
    no_nn = max(idxs)  # The kdtree querry uses this value as a standin for "no more neighbors"
    idxs = list(filter(lambda x: x != no_nn, idxs))
    new_idxs = []
    for i in idxs:
        # Relocate the correct index in the dataframe reference (the tree only indexes A sites)
        x, y = kdtree.data[i]
        match = df.loc[(df["x_fit"] == x) & (df["y_fit"] == y)]
        if len(match) != 1:
            raise RuntimeError("Something went wrong!")
        new_idxs.append(match.index[0])
    # Sometimes we find no neighbors if the column was fit poorly; just throw it out
    if new_idxs == []:
        new_idxs = None
    return new_idxs


def _get_ab_ratios(row, df):
    """Helper function to get column intensity ratios (B/A)."""
    if row["neighbors"] is not None:
        neighbor_intensities = []
        for i in row["neighbors"]:
            neighbor_intensities.append(df["total_col_int"].iloc[i])
        return row["total_col_int"] / np.mean(neighbor_intensities)


def _reject_outliers(row, df, outlier_scale):
    # Default outlier scale only rejects especially egregious outliers
    rmed = np.nanmedian(df["int_ratio"])
    rmad = stats.median_abs_deviation(df["int_ratio"], nan_policy="omit")
    if abs(row["int_ratio"] - rmed) > outlier_scale*rmad:
        return np.NaN
    else:
        return row["int_ratio"]


def thickness_est(x, a, c):
    # TODO: This is currently univariate, but can easily be made to depend on (x,y) points
    return a*x + c


def _thickness_modeler(points):
    # Take a series of 3D points (x, y, thickness) and model a thickness surface for the image
    # First separate out the x, y data from thickness data to effectively feed the optimizer
    # TODO: This is currently univariate, but can easily be made to depend on (x,y) points
    coords = np.array([x for x, _ in points])
    thicks = [t for _, t in points]
    return optimize.curve_fit(thickness_est, coords, thicks)


def ratio_est(t, a1, a2, a3, a4):
    return a1*t**3 + a2*t**2 + a3*t + a4


def _ratio_modeler(thicks, actual_ratios):
    return optimize.curve_fit(ratio_est, thicks, actual_ratios)


def get_ab_normalized_intensties(df, maxdist=36, n=4, a_elem="Ba"):
    # This is probably not the right way to normalize the intensities
    # I'm just savingthis code in a function in case I want to use it for something later
    # KDTree for fast NN lookup
    a_site_tree = spatial.KDTree(df.loc[df["elem"] == a_elem, "x_fit":"y_fit"])
    # Create the neighbors column by finding the four nearest A sites
    df["neighbors"] = df.apply(lambda row:
                               a_site_tree.query((row["x_fit"], row["y_fit"]),
                                                 distance_upper_bound=maxdist, k=n+1),
                               axis=1)
    # The kdtree query leaves us with unwieldy formatting for the neighborhood
    # So we need to fix that...  also assign None to the A sites so we don't compute on them
    df.reset_index(drop=True, inplace=True)
    df["neighbors"] = df.apply(lambda row: None if row["elem"] == a_elem else
                               _fmt_neighbors(row, df, a_site_tree), axis=1)
    # Compute intensity ratios for each B site row, and toss outliers
    df["int_ratio"] = df.apply(lambda row: _get_ab_ratios(row, df), axis=1)


def _get_near_neighbors(row, tree, n):
    # I couldn't come up with a simple formula for the neighbor indices, so instead I'll go back
    #  to the classic k-D Tree; it'll be on fraction coordinates multiplied by the basis.  It
    #  should be passed into this function, to avoid rebuilding it every time in O(nlogn)
    site = np.ndarray([row["u"], row["v"]]) @ lattice.a_2d
    near_neighbors = tree.query(site, k=n+1)
    near_neighbors = 
    return near_neighbors


def get_bb_normalized_intensities(row, df, tree, n=4):
    # Normalize based on a standard-score-like method on nearby B sites
    neighbors = _get_near_neighbors(row, tree, n)


def compute_neighborhood_stats(df, outlier_scale=10):
    fractional_coords = df.loc[:, "u":"v"].to_numpy()
    coords_in_A = fractional_coords @ lattice.a_2d
    btree = spatial.KDTree(coords_in_A)

    df["norm_int"] = df.apply(lambda row: get_bb_normalized_intensities(row, df, btree, n=8))
    df["norm_int"] = df.apply(lambda row: _reject_outliers(row, df,
                                                           outlier_scale), axis=1)
    # Compute the dispersion
    df["disp"] = df.apply(lambda row: (row["x_fit"] - row["x_ref"],
                                       row["y_fit"] - row["y_ref"]), axis=1)

    # Assign an estimated thickness to each site
    pts = [(135.4, 2), (1148.0, 16), (2160.6, 26)]
    optimized_thk_coefficients, _ = _thickness_modeler(pts)
    a, c = optimized_thk_coefficients
    df["est_thickness"] = df.apply(lambda row: thickness_est(row["x_fit"],
                                                             a, c), axis=1)

    # Discard the A sites, and any outliers, then drop irrelevant columns
    drop_idxs = df[np.isnan(df["int_ratio"])].index
    df.drop(drop_idxs, inplace=True)
    df.drop(["elem", "site_frac", "x", "y", "weight", "neighbors"],
            axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Assign each site an expected ratio based on the thickness
    thicknesses = list(df["est_thickness"])
    ratios = list(df["int_ratio"])
    optimized_ratio_coefficients, _ = _ratio_modeler(thicknesses, ratios)
    print(optimized_ratio_coefficients)
    a1, a2, a3, a4 = optimized_ratio_coefficients
    df["est_ratio"] = df.apply(lambda row: ratio_est(row["est_thickness"], a1, a2, a3, a4), axis=1)

    # Assign ratio delta (large positive values indicate columns more intense than expected)
    df["ratio_delta"] = df.apply(lambda row: row["int_ratio"] - row["est_ratio"], axis=1)


compute_neighborhood_stats(bframe)

# %% PYSAL ANALYSIS -- GLOBAL MORAN'S I


def make_adjlist(df, zone, a_2d=None, a_elem="Ba"):
    # Build adjacency list for defining weights object in a form digestable by pySAL
    """
    Implemented zones:
        100 == "100" == 110 == "110" == "rook" (normal rook adjacency)
        "queen" adds on diagonal adjacency to rook
        120 == "120" adds on in-row 2nd NNs to queen
    """
    adjlist = {}
    zones = [100, "100", 110, "110", "rook", "queen", 120, "120"]
    for idx, row in df.iterrows():
        if zone in zones:
            uminus_neighbor = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"])]
            uplus_neighbor = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"])]
            vminus_neighbor = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] - 1)]
            vplus_neighbor = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] + 1)]

            nlist = [n.index[0] for n in [uminus_neighbor, uplus_neighbor,
                                          vminus_neighbor, vplus_neighbor] if n.size > 0]

            if zone in zones[5:]:  # TODO: just slicing like this is probably not very sustainable
                # Consider finding a more elegant way of defining adjacency (maybe graphically?)
                # Extra neighbors on the diagonals
                uv_pp = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"] + 1)]
                uv_mm = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"] - 1)]
                uv_pm = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"] - 1)]
                uv_mp = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"] + 1)]
                nlist += [n.index[0] for n in [uv_pp, uv_mm, uv_pm, uv_mp] if n.size > 0]

                if zone in zones[6:]:
                    if a_2d is None:
                        raise TypeError("An array must be passed into a_2d for this zone")

                    # Need to get 2nd NNs in the B row ==> extra neighbors on shorter of u or v
                    if np.linalg.norm(a_2d[0]) < np.linalg.norm(a_2d[1]):  # u is shorter
                        nn2_up = df.loc[(df["u"] == row["u"] + 2) & (df["v"] == row["v"])]
                        nn2_dn = df.loc[(df["u"] == row["u"] - 2) & (df["v"] == row["v"])]
                    else:  # v is shorter
                        nn2_up = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] + 2)]
                        nn2_dn = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] - 2)]
                    nlist += [n.index[0] for n in [nn2_up, nn2_dn] if n.size > 0]

            nlist = tuple(nlist)
            adjlist[idx] = nlist
        else:
            raise NotImplementedError("Passed zone has no implemented neighbor rule")
    return adjlist


adjlist = make_adjlist(bframe, "queen", a_2d=lattice.a_2d)

# Create the pySAL weights object
w = weights.W(adjlist)
w.transform = "r"  # Transform to row-standard form

ratios = [r for r in bframe["ratio_delta"]]

mor = esda.moran.Moran(ratios, w, permutations=10000)
print(f"Global Moran's I:     {round(mor.I, 5)}\n"
      f"Expected I for CSR:  {round(mor.EI_sim, 5)}\n"
      f"p-value (10k perms.): {round(mor.p_sim, 5)}")

# %% PYSAL ANALYSIS -- LOCAL MORAN'S I

norm = Normalize(vmin=min(bframe["ratio_delta"]),
                 vmax=max(bframe["ratio_delta"]), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

mor_loc = esda.Moran_Local(ratios, w, permutations=10000)

bframe["moran_I"] = mor_loc.Is
bframe["moran_p"] = mor_loc.p_sim
bframe["moran_quadrant"] = mor_loc.q


def colorize_cluster(ps, quads, p_bands=[0.001, 0.01, 0.05], tints=[0, 1/3, 2/3]):
    # RGB color coding for clusters and outliers
    hh_clst_color = (41/255, 235/255, 0)   # Green
    lh_otlr_color = (240/255, 0, 180/255)  # Purple
    ll_clst_color = (0, 113/255, 235/255)  # Blue
    hl_otlr_color = (245/255, 155/255, 0)  # Orange

    colors = [hh_clst_color, lh_otlr_color, ll_clst_color, hl_otlr_color]  # ORDER MATTERS
    color_arr = [colors[q-1] for q in quads]

    # If the simulated p value is significant, make the point visible
    alpha_arr = [(1,) if p <= max(p_bands) else (0,) for p in ps]

    # Decrease saturation based on significance band
    color_arr = [(c[0] + (1 - c[0]) * next(i * 1/len(p_bands)
                                           for i, v in enumerate(p_bands) if v > p),
                  c[1] + (1 - c[1]) * next(i * 1/len(p_bands)
                                           for i, v in enumerate(p_bands) if v > p),
                  c[2] + (1 - c[2]) * next(i * 1/len(p_bands)
                                           for i, v in enumerate(p_bands) if v > p))
                 if p <= max(p_bands) else c for c, p in zip(color_arr, ps)]

    z = zip(color_arr, alpha_arr)
    rgbas = [to_rgba(rgb + a) for rgb, a in z]
    return rgbas


fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.17, 1]})
axs[0].imshow(image_cropped, cmap="gray")
axs[1].imshow(image_cropped, cmap="gray")

xs, ys = bframe["x_fit"], bframe["y_fit"]
s = 70
axs[0].scatter(xs, ys, color=mapper.to_rgba(ratios), s=s, linewidths=0)

ps = [p for p in bframe["moran_p"]]  # REMEMBER: These are *pseudo* p-values
quads = [q for q in bframe["moran_quadrant"]]
axs[1].scatter(xs, ys, color=colorize_cluster(ps, quads), s=s, linewidths=0)

axs[0].axis("off")
axs[1].axis("off")
fig.colorbar(mapper, ax=axs[0], label="B/A Intensity Ratio", location="left", fraction=0.04)

# %% THICKNESS SERIES TESTING

# Select the correct B sites (the ones centered in full B site columns)
# Then assign them their correct thicknesses

thickness_list = [1.6332, 2.4498, 2.8581, 3.2664, 3.6747, 4.4913, 4.8996, 5.3079, 5.7162, 6.5328,
                  6.9411, 7.3494, 7.7577, 8.5743, 8.9826, 9.3909, 9.7992, 10.6158, 11.0241,
                  11.4324, 11.8407, 12.6573, 13.0656, 13.4739, 13.8822, 14.6988, 15.1071, 15.5154,
                  15.9237, 16.7403, 17.1486, 17.5569, 17.9652, 18.7818, 19.1901, 19.5984, 20.0067,
                  20.8233, 21.6399, 22.8648, 23.6814, 24.9063, 25.7229, 26.9478]


def assign_sites(df, thickness_list, image_width=3, image_height=None, reps=4):
    if image_height is None:
        image_height = image_width
    df_new = deepcopy(df)
    # Min_u, max_v should be the top-leftmost B site
    df_new["selected"] = df_new.apply(lambda _: False, axis=1)
    df_new["est_thickness"] = df_new.apply(lambda _: False, axis=1)
    min_u, max_v = min(df_new["u"]), max(df_new["v"])
    initial_uvs = (min_u + 1, max_v - 1)
    for i, t in enumerate(thickness_list):
        u = initial_uvs[0] + i*image_width
        for j in range(reps):
            v = initial_uvs[1] - j*image_height
            df_new.loc[(df_new["u"] == u) & (df_new["v"] == v), "selected"] = True
            df_new.loc[(df_new["u"] == u) & (df_new["v"] == v), "est_thickness"] = t

    return df_new[df_new["selected"]]


bframe_subset = assign_sites(bframe, thickness_list)

ratio_dict = {}
stdev_dict = {}
for _, row in bframe_subset.iterrows():
    ratio_dict[row["est_thickness"]] = []
for _, row in bframe_subset.iterrows():
    ratio_dict[row["est_thickness"]].append(row["int_ratio"])
for key, values in ratio_dict.items():
    stdev_dict[key] = np.std(values)
    ratio_dict[key] = np.mean(values)

plt.plot(ratio_dict.keys(), ratio_dict.values(), "r-")
plt.fill_between(ratio_dict.keys(), [r-s for r, s in zip(ratio_dict.values(),
                                                         stdev_dict.values())],
                 [r+s for r, s in zip(ratio_dict.values(), stdev_dict.values())],
                 color='#ff000080')
