"""
1. Read in a STEM image
2. Create a model of the structure in the image
3. Fit the atom columns using SingleOrigin
4. Perform statistical analysis!
"""

# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

import SingleOrigin as so

import quickcrop.quickcrop as qc
from scipy import spatial, stats, optimize, ndimage, signal
from copy import deepcopy
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgba

import esda
from libpysal import weights

# %% CUSTOM FUNCTIONS


def divide_image_frequencies(img, s, show_gauss=False, show_images=False):
    """
    Parameters
    ----------
    img : 2D array of scalar
        A 2D array of scalar value representing an image
    s : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are
        given for each axis as a sequence, or as a single number, in which case it is equal for
        all axes.
    show_gauss : bool
        If True, plot the Gaussian kernel (using pyplot) before returning the frequency divided
        images.  The kernel is plotted in a square with side length img.shape[0].  The default
        value is False
    show_images : bool
        If True, plot the low and high frequency images before returning.  Default is False

    Returns
    -------
    lowpass : 2D array of float
        The lowpass image (the direct result of theGaussian filter)
    highpass : 2D array of float
        The highpass image (equivalen to img - lowpass)

    """
    if show_gauss:
        g1d = signal.windows.gaussian(img.shape[0], s)
        g2d = np.outer(g1d, g1d)
        plt.imshow(g2d, cmap="magma")

    lowpass = ndimage.gaussian_filter(img, sigma=s, mode="reflect")
    highpass = img - lowpass

    if show_images:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(lowpass, cmap="magma")
        axs[1].imshow(highpass, cmap="magma")
    return lowpass, highpass


def grow_tree(df, a2d, kind):
    """
    Parameters
    ----------
    df : DataFrame
    kind : String or list of strings or None
        String(s) representing the kind of atom columns which are valid neighbors; matches against
        the "elem" column in df.  A value of None matches all kinds of columns

    Returns
    -------
    numpy kdtree (2D) that can be used for fast near-neighbor searching.
    """
    fractional_coords = df.loc[df["elem"] == kind, "u":"v"].to_numpy()
    coords_in_A = fractional_coords @ a2d
    tree = spatial.KDTree(coords_in_A)
    return tree


def get_near_neighbors(row, df, a2d, tree, n, kind):
    """
    Parameters
    ----------
    row : DataFrame row
        The row representing the site that's currently being searched around
    df : DataFrame
        The DataFrame represening the sites fitted in the image
    tree : kdtree
        A numpy kdtree used to search for near neighbors given the coordinates in row
    n : int
        The number of near neighbors to search for
    kind : string or list of strings
        String(s) representing the kind of atom columns which are valid neighbors; matches against
        the "elem" column in df.  A value of None matches all kinds of columns

    Raises
    ------
    RuntimeError
        Happens if a neighbor is identified in the tree but can't be matched to an entry in df

    Returns
    -------
    list of ints
        The indices of the near neighbors of a given site, as indexed in df
    """
    u, v = row["u"], row["v"]
    site = np.asarray((u, v)) @ a2d
    near_neighbors = tree.query(site, k=n+1)

    # The kdtree query returns an unwieldy format for the neighborhood, which we need to fix
    no_nn = tree.n  # This k-D tree implementation uses self.n to indicate a missing neighbor
    idxs = list(filter(lambda x: x != no_nn, near_neighbors[1]))
    neighborhood = []
    for i in idxs:
        x, y = tree.data[i]
        a2d_inv = np.linalg.inv(a2d)
        u_match, v_match = np.asarray((x, y)) @ a2d_inv
        match = df.loc[(np.isclose(df["u"], u_match)) & (np.isclose(df["v"], v_match))]
        if len(match) != 1:
            raise RuntimeError("No near-neighbor (u, v) matches found; this shouldn't happen!")
        neighborhood.append(match.index[0])
    # Sometimes we find no neighbors if the column was fit poorly; just throw it out
    if neighborhood == []:
        neighborhood = None
    return neighborhood


def _test_outliers(x, med, mad, outlier_scale):
    if abs(x - med) > outlier_scale*mad:
        return np.NaN
    else:
        return x


def reject_outliers(df, outlier_scale=10):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    outlier_scale : float, optional
        Determines what counts as an outlier.  Rows are outliers if the difference between the
        row's dispersion and the median dispersion in df is greater than `outlier_scale * MAD`
        where `MAD` is the median absolute deviation of the dispersions in df.  The default of 10
        only rejects *extremely egregious* outliers

    Returns
    -------
    None, but mutates df by dropping outlier rows and resetting the indices of df
    """
    for col in ["disp_mag", "norm_int"]:
        rmed = np.median(df[col])
        rmad = stats.median_abs_deviation(df[col])

        df[col] = df.apply(lambda row: _test_outliers(row[col], rmed, rmad, outlier_scale),
                           axis=1)

        outlier_count = df[col].isna().sum()
        if outlier_count != 0:
            print(f"Dropping {outlier_count} outlier values based on values in '{col}'.")
            df.drop(df.loc[np.isnan(df[col])].index, inplace=True)
            df.reset_index(drop=True, inplace=True)


def calculate_dispersion(df):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    Returns
    -------
    None, but mutates df by adding a column "disp" representing dispersion (vector displacement
    from the expected location based on a perfect lattice)
    """
    df["disp"] = df.apply(lambda row: (row["x_fit"] - row["x_ref"],
                                       row["y_fit"] - row["y_ref"]), axis=1)
    df["disp_mag"] = df.apply(lambda row: np.linalg.norm(row["disp"]), axis=1)


def drop_elements(df, e):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    elem : list of string
        The elements to be removed from df, matched against the "elem" column

    Returns
    -------
    None, but mutates df by removing rows that match any of the elements in elem, and then
    resetting the indices
    """
    df.drop(df.loc[df["elem"].isin(e)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _ratio(row, df):
    site_intensity = row["total_col_int"]
    neighbor_intensities = [df.iloc[i]["total_col_int"] for i in row["neighborhood"]]
    return site_intensity / np.mean(neighbor_intensities)


def _standard_score(row, df):
    site_intensity = row["total_col_int"]
    neighbor_intensities = [df.iloc[i]["total_col_int"] for i in row["neighborhood"]]
    mean, stdev = np.mean(neighbor_intensities), np.std(neighbor_intensities)
    return (site_intensity - mean) / stdev


def _off(row):
    return row["total_col_int"] 


def normalize_intensity(df, a2d, n=4, method="ratio", kind=None):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    a2d : 2x2 array of floats
        The transformation matrix to go from (u, v) coordinates into Angstroms
    n : int, optional
        The number of neighboring sites to normalize against. The default is 4
    method : string or None, optional
        The normalization method to use.  Available methods are "ratio", "standard_score", and
        "off"/None.  The default is "ratio"
    kind : string or None
        String(s) representing the kind of atom columns which are valid neighbors; matches against
        the "elem" column in df.  A value of None matches all kinds of columns

    Raises
    ------
    NotImplementedError
        Occurs if a string not represnting an implemented method is passed in the method parameter

    Returns
    -------
    None, but mutates df by adding columns "neighborhood" and "norm_int" representing the n near
    neighbord of the kind specified, and normalized column intensity
    """
    tree = grow_tree(df, a2d, kind)
    df["neighborhood"] = df.apply(lambda row:
                                  get_near_neighbors(row, df, a2d, tree, n, kind), axis=1)

    match method:
        case "ratio":
            df["norm_int"] = df.apply(lambda row: _ratio(row, df), axis=1)
        case "standard score" | "score":
            df["norm_int"] = df.apply(lambda row: _standard_score(row, df), axis=1)
        case "off" | None:
            df["norm_int"] = df.apply(lambda row: _off(row), axis=1)
        case _:
            raise NotImplementedError("The specified normalization method" +
                                      " has not been implemented.")


def _rookzone(df, row):
    u_m = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"])]
    u_p = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"])]
    v_m = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] - 1)]
    v_p = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] + 1)]

    return [n.index[0] for n in [u_m, u_p, v_m, v_p] if n.size > 0]


def _bishopzone(df, row):
    uv_pp = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"] + 1)]
    uv_mm = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"] - 1)]
    uv_pm = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"] - 1)]
    uv_mp = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"] + 1)]

    return [n.index[0] for n in [uv_pp, uv_mm, uv_pm, uv_mp] if n.size > 0]


def _120zone(df, row, a2d):
    # Need to get 2nd NNs in the B row ==> extra neighbors on shorter of u or v
    if np.linalg.norm(a2d[0]) < np.linalg.norm(a2d[1]):  # u is shorter
        nn2_up = df.loc[(df["u"] == row["u"] + 2) & (df["v"] == row["v"])]
        nn2_dn = df.loc[(df["u"] == row["u"] - 2) & (df["v"] == row["v"])]
    else:  # v is shorter
        nn2_up = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] + 2)]
        nn2_dn = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] - 2)]

    return [n.index[0] for n in [nn2_up, nn2_dn] if n.size > 0]


def make_weights(df, adj_type, a2d=None):
    """
    Build adjacency list for defining weights object in a form digestable by pySAL

    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image
    adj_type : string
        Implemented zones:
            "rook" counts only the four nearest neighbors as adjacent
            "queen" and "king" are synonymous and add on diagonal adjacency to rook
            "bishop" counts only sites that would be in queen but not in rook
            "120" adds on in-row 2nd NNs to queen, and requires a2d to be passed
    a2d : 2x2 array of floats, optional
        Used to transform from uv coordinates to Angstroms, required for zone 120

    Raises
    ------
    RuntimeError
        Raised when attempting to use zone 120 without passing a2d
    NotImplementedError
        Raised when trying to use a zone which is not implemented

    Returns
    -------
    asdf
    """
    adjlist = {}
    for idx, row in df.iterrows():
        neighbor_list = []
        match adj_type:
            case "rook":
                neighbor_list += _rookzone(df, row)
            case "bishop":
                neighbor_list += _bishopzone(df, row)
            case "queen" | "king":
                neighbor_list += _rookzone(df, row)
                neighbor_list += _bishopzone(df, row)
            case "120":
                if a2d is None:
                    raise RuntimeError("a2d must be passed as an argument to use adj_type 120")
                neighbor_list += _rookzone(df, row)
                neighbor_list += _bishopzone(df, row)
                neighbor_list += _120zone(df, row, a2d)
            case _:
                raise NotImplementedError("The requested adjacency type has not been implemented")
        neighbor_list = tuple(neighbor_list)
        adjlist[idx] = neighbor_list

    w = weights.W(adjlist)  # Create the pySAL weights object from the adjacency list
    w.transform = "r"  # Transform to row-standard form
    return w


def moran_global(df, adj_type, a2d, p=10000, printstats=True):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image
    adj_type : string
        The kind of adjacency rule to use.  See the documentation for make_weights
    a2d : 2x2 array of floats
        Used to transform from uv coordinates to Angstroms
    p : int, optional
        The number of permutations to use for testing against CSR. The default is 10000
    printstats : bool
        Whether or not to print some statistics related to the global Moran's I test

    Returns
    -------
    mor : esda.moran.Moran
        Object containing the information used in the Moran's I analysis
    """
    w = make_weights(df, adj_type, a2d)
    nints = [ni for ni in df["norm_int"]]
    mor = esda.moran.Moran(nints, w, permutations=p)
    
    if printstats:
        print(f"Global Moran's I:     {round(mor.I, 5)}\n"
              f"Expected I for CSR:  {round(mor.EI_sim, 5)}\n"
              f"p-value (10k perms.): {round(mor.p_sim, 5)}")
        if mor.p_sim < 0.05:
            print("Observed distribution not consistent with complete spatial randomness")
        else:
            print("Observed distribution is consistent with complete spatial randomness")

    return mor


def moran_local(df, adj_type, a2d, p=10000):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image
    adj_type : string
        The kind of adjacency rule to use.  See the documentation for make_weights
    a2d : 2x2 array of floats
        Used to transform from uv coordinates to Angstroms
    p : int, optional
        The number of permutations to use for testing against CSR. The default is 10000

    Returns
    -------
    mor : esda.moran.Moran_Local
        Object containing the information used in the local Moran's I analysis
    """
    w = make_weights(df, adj_type, a2d)
    nints = [ni for ni in df["norm_int"]]
    mor = esda.moran.Moran_Local(nints, w, permutations=p)
    return mor


def add_stats_to_frame(df, mor):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame to which the stats should be added
    mor : esda.moran.Moran_Local
        The object containing the local Moran statistics to be added to the frame

    Returns
    -------
    None, but will of course mutate df
    """
    df["moran_I"] = mor.Is
    df["moran_p"] = mor.p_sim
    df["moran_quad"] = mor.q


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


def plot_local_clusters(df, img, intensity_colormap=cm.viridis,
                        image_colormap="gray", s=70):
    norm = Normalize(vmin=min(df["norm_int"]),
                     vmax=max(df["norm_int"]), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=intensity_colormap)

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.17, 1]})
    axs[0].imshow(img, cmap=image_colormap)
    axs[1].imshow(img, cmap=image_colormap)

    xs, ys = df["x_fit"], df["y_fit"]
    axs[0].scatter(xs, ys, color=mapper.to_rgba([ni for ni in df["norm_int"]]), s=s, linewidths=0)

    ps = [p for p in df["moran_p"]]  # REMEMBER: These are *pseudo* p-values
    quads = [q for q in df["moran_quad"]]
    axs[1].scatter(xs, ys, color=colorize_cluster(ps, quads), s=s, linewidths=0)

    axs[0].axis("off")
    axs[1].axis("off")
    fig.colorbar(mapper, ax=axs[0], label="Normalized Intensity", location="left", fraction=0.04)


# %% LOAD DRIFT CORRECTED IMAGE

IMPORT_PATH = r"E:\Users\Charles\bzt_data_100_20221118"
IMPORT_NAME = "11-16-2022_14.56.19_HAADFdrift_corr_HAADF_NOGRD.tif"
image_cropped = np.array(tif.imread(os.path.join(IMPORT_PATH, IMPORT_NAME)))
image_cropped = qc.gui_crop(image_cropped)
image_cropped = so.image_norm(image_cropped)

_, image_cropped = divide_image_frequencies(image_cropped, s=350, show_images=True)
image_cropped = so.image_norm(image_cropped)

# %% SINGLEORIGIN INITIALIZATION

CIF_PATH = r"E:\Users\Charles\BZT.cif"

za = [1, 0, 0]  # Zone axis direction
a1 = [0, 1, 0]  # Apparent horizontal axis in projection
a2 = [0, 0, 1]  # Most vertical axis in projection

uc = so.UnitCell(CIF_PATH, origin_shift=[0, 0, 0])
uc.atoms.replace("Ti/O", "Ti/Zr", inplace=True)

# Ignore light elements for HAADF
uc.project_zone_axis(za, a1, a2, ignore_elements=["O"])
uc.combine_prox_cols(toler=1e-2)
# uc.plot_unit_cell()  # Uncomment and check this output if changing the u.c.

hr_img = so.HRImage(image_cropped)
lattice = hr_img.add_lattice("BZT", uc)

# Tidy namespace
del za, a1, a2, uc, IMPORT_PATH, IMPORT_NAME, CIF_PATH

# %% PREPARE FOR FITTING
# NOTE: There are a couple of steps here that require interaction and will time out if ignored

# If some FFT peaks are weak or absent (such as forbidden reflections),
#  specify the order of the first peak that is clearly visible
lattice.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)

lattice.get_roi_mask_std(r=15, buffer=20, thresh=0.25, show_mask=True)
# lattice.roi_mask = np.ones(image_cropped.shape)  # For pre-cropped images

lattice.define_reference_lattice()

# %% ATOM COLUMN FITTING

# TODO: parallelization is broken in my current python environment; fix this at some point!
lattice.fit_atom_columns(buffer=10, local_thresh_factor=0.1, use_background_param=False,
                         use_bounds=True, use_circ_gauss=False, parallelize=True)

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

# hr_img.plot_disp_vects()

# %% CREATE FRAME

# A copy of the at_cols dataframe of the lattice object, for convenience
frame = deepcopy(lattice.at_cols)

frame.drop(["site_frac", "x", "y", "weight"], axis=1, inplace=True)  # We don't need these cols
frame.reset_index(drop=True, inplace=True)

normalize_intensity(frame, lattice.a_2d, n=24, method="score", kind="Ti/Zr")
calculate_dispersion(frame)
drop_elements(frame, ["Ba"])
reject_outliers(frame)

# %% PYSAL ANALYSIS -- GLOBAL MORAN'S I

mor_glb = moran_global(frame, "queen", lattice.a_2d, p=10000, printstats=True)

# %% PYSAL ANALYSIS -- LOCAL MORAN'S I

mor_lcl = moran_local(frame, "queen", lattice.a_2d, p=10000)
add_stats_to_frame(frame, mor_lcl)

plot_local_clusters(frame, image_cropped)

# %% THICKNESS SERIES TESTING

# I probably don't need this anymore, but I'll keep it for now

# Select the correct B sites (the ones centered in full B site columns)
# Then assign them their correct thicknesses

# thickness_list = [1.6332, 2.4498, 2.8581, 3.2664, 3.6747, 4.4913, 4.8996, 5.3079, 5.7162, 6.5328,
#                   6.9411, 7.3494, 7.7577, 8.5743, 8.9826, 9.3909, 9.7992, 10.6158, 11.0241,
#                   11.4324, 11.8407, 12.6573, 13.0656, 13.4739, 13.8822, 14.6988, 15.1071, 15.5154,
#                   15.9237, 16.7403, 17.1486, 17.5569, 17.9652, 18.7818, 19.1901, 19.5984, 20.0067,
#                   20.8233, 21.6399, 22.8648, 23.6814, 24.9063, 25.7229, 26.9478]


# def assign_sites(df, thickness_list, image_width=3, image_height=None, reps=4):
#     if image_height is None:
#         image_height = image_width
#     df_new = deepcopy(df)
#     # Min_u, max_v should be the top-leftmost B site
#     df_new["selected"] = df_new.apply(lambda _: False, axis=1)
#     df_new["est_thickness"] = df_new.apply(lambda _: False, axis=1)
#     min_u, max_v = min(df_new["u"]), max(df_new["v"])
#     initial_uvs = (min_u + 1, max_v - 1)
#     for i, t in enumerate(thickness_list):
#         u = initial_uvs[0] + i*image_width
#         for j in range(reps):
#             v = initial_uvs[1] - j*image_height
#             df_new.loc[(df_new["u"] == u) & (df_new["v"] == v), "selected"] = True
#             df_new.loc[(df_new["u"] == u) & (df_new["v"] == v), "est_thickness"] = t

#     return df_new[df_new["selected"]]


# bframe_subset = assign_sites(bframe, thickness_list)

# ratio_dict = {}
# stdev_dict = {}
# for _, row in bframe_subset.iterrows():
#     ratio_dict[row["est_thickness"]] = []
# for _, row in bframe_subset.iterrows():
#     ratio_dict[row["est_thickness"]].append(row["int_ratio"])
# for key, values in ratio_dict.items():
#     stdev_dict[key] = np.std(values)
#     ratio_dict[key] = np.mean(values)

# plt.plot(ratio_dict.keys(), ratio_dict.values(), "r-")
# plt.fill_between(ratio_dict.keys(), [r-s for r, s in zip(ratio_dict.values(),
#                                                          stdev_dict.values())],
#                  [r+s for r, s in zip(ratio_dict.values(), stdev_dict.values())],
#                  color='#ff000080')
