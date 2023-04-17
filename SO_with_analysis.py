"""
1. Read in a STEM image
2. Create a model of the structure in the image
3. Fit the atom columns using SingleOrigin
4. Perform statistical analysis!
"""

# %% IMPORTS
import os
import typing
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

import SingleOrigin as so

#import quickcrop.quickcrop as qc
from scipy import spatial, stats, optimize, ndimage, signal
from copy import deepcopy

from matplotlib import cm
from matplotlib.colors import Normalize, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib import gridspec
import cmasher

import esda
from libpysal import weights
from random import shuffle

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


def get_near_neighbors(row, df, a2d, tree, n):
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


def reject_outliers(df, mode, outlier_scale=10):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    mode : str
        Should be one of "total_col_int" or "disp_mag"; determines which column in df is used when
        checking for outlier status
    outlier_scale : float, optional
        Determines what counts as an outlier.  Rows are outliers if the difference between the
        row's dispersion and the median dispersion in df is greater than `outlier_scale * MAD`
        where `MAD` is the median absolute deviation of the dispersions in df.  The default of 10
        only rejects *extremely egregious* outliers

    Returns
    -------
    None, but mutates df by dropping outlier rows and resetting the indices of df
    """
    rmed = np.median(df[mode])
    rmad = stats.median_abs_deviation(df[mode])

    df[mode] = df.apply(lambda row: _test_outliers(row[mode], rmed, rmad, outlier_scale), axis=1)

    outlier_count = df[mode].isna().sum()
    if outlier_count != 0:
        print(f"Dropping {outlier_count} outlier site{'s'[:outlier_count^1]}" +
              f" based on values in '{mode}'.")
        df.drop(df.loc[np.isnan(df[mode])].index, inplace=True)
        df.reset_index(drop=True, inplace=True)


def _dot_disp(row, df):
    neighbor_disps = list(map(np.array, [df.iloc[i]["disp"] for i in row["neighborhood"]]))
    mean_neighbor_disp = np.sum(neighbor_disps, axis=0)/len(neighbor_disps)
    return np.dot(row["disp"], mean_neighbor_disp)


def disp_calc(df, kind=None, normalize=False):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    kind : str
        The kind of column to normalize to, matching against `df["elem"]`, or "all".  The default
        is None
    normalize : bool
        Whether or not to add a column with (globally) normalized values for `dot_disp`.  The
        default is False
    Returns
    -------
    None, but mutates df by adding a column "disp" representing dispersion (vector displacement
    from the expected location based on a perfect lattice)
    """
    df["disp"] = df.apply(lambda row: (row["x_fit"] - row["x_ref"],
                                       row["y_fit"] - row["y_ref"]), axis=1)
    df["dot_disp"] = df.apply(lambda row: _dot_disp(row, df), axis=1)

    if normalize:
        if kind is None:
            raise RuntimeError("Must set `kind` if using `normalize`")
        if kind == "all":
            ddmin, ddmax = min(df["dot_disp"]), max(df["dot_disp"])
        else:
            ddmin = min(df.loc[df["elem"] == kind, "dot_disp"])
            ddmax = max(df.loc[df["elem"] == kind, "dot_disp"])
        df["dot_normal_disp"] = df.apply(lambda row:
                                         (row["dot_disp"] - ddmin)/(ddmax - ddmin), axis=1)


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


def _global_minmax(row, df, kind):
    site_intensity = row["total_col_int"]
    if kind is None:
        min_int, max_int = min(df["total_col_int"]), max(df["total_col_int"])
    else:
        min_int = min(df.loc[df["elem"] == kind, "total_col_int"])
        max_int = max(df.loc[df["elem"] == kind, "total_col_int"])
    return (site_intensity - min_int) / (max_int - min_int)


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
        The normalization method to use.  Available methods are "ratio", "standard_score",
        "off"/None, and "global_minmax".  The default is "ratio"
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
                                  get_near_neighbors(row, df, a2d, tree, n), axis=1)

    match method:
        case "ratio":
            df["norm_int"] = df.apply(lambda row: _ratio(row, df), axis=1)
        case "standard score" | "score":
            df["norm_int"] = df.apply(lambda row: _standard_score(row, df), axis=1)
        case "off" | None:
            df["norm_int"] = df.apply(lambda row: _off(row), axis=1)
        case "global_minmax":
            df["norm_int"] = df.apply(lambda row: _global_minmax(row, df, kind), axis=1)
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
        # The indices need to be strings to avoid a bug in pySAL esda.geary_local_mv where if the
        # indices are numeric, they get summed into the statistic (!), destroying any real signal
        neighbor_list = tuple(map(str, neighbor_list))
        adjlist[str(idx)] = neighbor_list

    w = weights.W(adjlist, silence_warnings=True)
    w.transform = "r"  # Transform to row-standard form
    return w


def _sink_islands(df, w):
    """
    Parameters
    ----------
    df : DataFrame
        The frame containing sites which are islands.
        Warning: this will mutate the frame by dropping island rows!
    w : libpysal.weights
        Weights object defining adjacency: this is the object that knows which sites are islands
    -------
    w : libpysal.weights
        Weights object with the islands removed

    """
    islands = set(w.islands)  # Having this be a set makes everything simpler and faster
    print(f"Dropping {len(islands)} outlier site{'s'[:len(islands)^1]} based on 'w.islands'")
    w = weights.w_subset(w, set(w.id_order) - islands)

    df.drop(index=islands, inplace=True)
    return w


def get_stats(df, adj_type, a2d, kind, columns, p=10000, printstats=True):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: if w contains islands, this will mutate df by removing them!
    adj_type : string
        The kind of adjacency rule to use.  See the documentation for `make_weights`.
    a2d : 2x2 array of floats
        Used to transform from uv coordinates to Angstroms.
    kind : string
        The kind of statistical test to perform.  Options are: "moran_global", "moran_local",
        "moran_global_bivariate", "moran_local_bivariate", "geary_global", "geary_local",
        and "geary_local_multivariate".
    columns : string or iterable of strings
        String(s) representing the column(s) in df to run the test on.  Note that not all
        values of `kind` are compatible with an arbitrary amount of strings!  Specifically:
        "moran_global", "moran_local", "geary_global", and "geary_local" require a single string
        (or a length one iterable of strings); "moran_global_bivariate" and "moran_local_bivariate"
        require an iterable containing exactly two strings; and "geary_local_multivariate" requires
        an iterable containing at least two strings.

        BEWARE: for the bivariate Moran methods, the order of the two strings in this parameter
        will change the meaning of the returned statistic!  For `columns=["a", "b"]`, the Moran
        statistic represents the inward influence upon the value of `a` at a given site of the
        values of `b` at neighboring sites.
    p : int, optional
        The number of permutations to use for testing against CSR. The default is 10000
    printstats : bool, optional
        Whether or not to print some statistics to the console. This setting only applies to global
        statistics (be they Moran or Geary).  The default is True.
    Returns
    -------
    stat_blob
        The statistics object returned by whatever pysal function ended up being called.  Can take
        on a variety of types depending on what kind of statistics are being used.
    """
    w = make_weights(df, adj_type, a2d)  # May contain islands, so sink them
    if len(w.islands) != 0:
        w = _sink_islands(df, w)

    if len(columns) == 0:
        raise RuntimeError("`columns` parameter must not be empty")

    # Match against the different kinds of statistical tests we might be performing
    match kind:
        # ====== # MORAN STATISTICS # ====== #
        case "moran_global" | "moran_local":
            # Valid columns types: string or iterable containing exactly one string
            if not (isinstance(columns, str) or (isinstance(columns, typing.Iterable)
                                                 and len(columns) == 1)):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")
            if not isinstance(columns, str):
                columns = columns[0]  # If the string is in another iterable, unwrap it

            vals = [val for val in df[columns]]

            if kind == "moran_global":
                stat_blob = esda.moran.Moran(vals, w, permutations=p)
                if printstats:
                    print(f"Global Moran's I:     {round(stat_blob.I, 5)}\n"
                          f"Expected I for CSR:  {round(stat_blob.EI_sim, 5)}\n"
                          f"p-value ({p} perms.): {round(stat_blob.p_sim, 5)}")
                    if stat_blob.p_sim < 0.05:
                        print("Observed distribution not consistent with CSR")
                    else:
                        print("Observed distribution consistent with CSR")

            elif kind == "moran_local":
                stat_blob = esda.moran.Moran_Local(vals, w, permutations=p)

        case "moran_global_bivariate" | "moran_local_bivariate":
            # Valid columns types: iterable containing exactly two strings
            if not (isinstance(columns, typing.Iterable) and len(columns) == 2):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")

            vals = [[val for val in df[col]] for col in columns]

            if kind == "moran_global_bivariate":
                stat_blob = esda.moran.Moran_BV(vals[0], vals[1], w, permutations=p)
                if printstats:
                    print(f"Global Moran's I:     {round(stat_blob.I, 5)}\n"
                          f"Expected I for CSR:  {round(stat_blob.EI_sim, 5)}\n"
                          f"p-value ({p} perms.): {round(stat_blob.p_sim, 5)}")
                    if stat_blob.p_sim < 0.05:
                        print("Observed distribution not consistent with CSR")
                    else:
                        print("Observed distribution consistent with CSR")

            elif kind == "moran_local_bivariate":
                stat_blob = esda.moran.Moran_Local_BV(vals[0], vals[1], w, permutations=p)

        # ====== # GEARY STATISTICS # ====== #
        case "geary_global" | "geary_local":
            # Valid columns types: string or iterable containing exactly one string
            if not (isinstance(columns, str) or (isinstance(columns, typing.Iterable)
                                                 and len(columns) == 1)):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")
            if not isinstance(columns, str):
                columns = columns[0]  # If the string is in another iterable, unwrap it

            vals = vals = [val for val in df[columns]]

            if kind == "geary_global":
                stat_blob = esda.Geary(vals, w, permutations=p)
                if printstats:
                    print(f"Global Gear's C':     {round(stat_blob.C, 5)}\n"
                          f"Expected C for CSR:  {round(stat_blob.EC_sim, 5)}\n"
                          f"p-value ({p} perms.): {round(stat_blob.p_sim, 5)}")
                    if stat_blob.p_sim < 0.05:
                        print("Observed distribution not consistent with CSR")
                    else:
                        print("Observed distribution consistent with CSR")

            elif kind == "geary_local":
                stat_blob = esda.Geary_Local(connectivity=w, labels=True, permutations=p)
                stat_blob.fit(vals)

        case "geary_local_multivariate":
            if not (isinstance(columns, typing.Iterable) and len(columns) >= 2):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")

            vals = [[val for val in df[col]] for col in columns]

            stat_blob = esda.Geary_Local_MV(connectivity=w, permutations=p)
            stat_blob.fit(vals)

        case _:
            raise NotImplementedError("The requested statistical test has not been implemented," +
                                      " or `kind` was an invalid string")

    return stat_blob


def add_stats_to_frame(df, sts, kind, sig=0.05):
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame to which the stats should be added
    sts : a variety of types depending on what kind of statistics are being used
        The object containing the local statistics to be added to the frame
    kind : string
        The kind of statistic being added (e.g. "moran" or "geary")
    Returns
    -------
    None, but will of course mutate df
    """
    df[f"{kind}_p"] = sts.p_sim
    if kind == "moran":
        df["moran_I"] = sts.Is
        df["moran_quad"] = sts.q
    if kind == "geary":
        df["geary_C"] = sts.localG
        if "labs" in sts.__dict__:
            df["geary_label"] = sts.labs
        # else:
        #     # pySAL's implementation of multivariate local Geary does not include labels
        #     # We need to do a bit of work to add them ourselves
        #     x = np.asarray(sts.variables).flatten()
        #     x_mean = np.mean(x)
        #     eij_mean = np.mean(sts.localG)
        #     labs = np.empty(len(x))
        #     locg_lt_eij = sts.localG < eij_mean
        #     p_leq_sig = sts.p_sim <= sig

        #     # Outliers
        #     labs[locg_lt_eij & (x > x_mean) & p_leq_sig] = "outlier"
        #     # Cluster members
        #     labs[locg_lt_eij & (x < x_mean) & p_leq_sig] = "cluster"
        #     # Other
        #     labs[(sts.localG > eij_mean) & p_leq_sig] = "other"
        #     # Non-significant
        #     labs[sts.p_sim > sig] = "non-significant"


def _moran_cluster_members(df, sig):
    core_members = {i for i, p in enumerate(df["moran_p"]) if (p <= sig)}
    members = set(core_members)
    for m in core_members:
        members.update(df.iloc[m]["neighborhood"])
    return members


def plot_moran_clusters(df, img, var_cmap=plt.get_cmap("cmr.amber"), image_cmap="bone", sig=0.05):
    # plt.style.use("dark_background")  # For testing, because it's easier to look at on a screen
    plt.style.use("seaborn-v0_8-colorblind")
    # Mappers for colorizing the voronoi plots
    norm = Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=var_cmap)

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.17, 1]})
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0)

    axs[0].set_title("Normalized Locally Correlated Displacement")
    fig.colorbar(mapper, ax=axs[0], location="left", fraction=0.04, pad=0.02)
    axs[1].set_title("Clusters")

    # Each set of axes gets an image
    for ax in axs:
        ax.imshow(img, cmap=image_cmap)
        ax.axis("off")
        # Set limits to keep them from being distorted by the voronoi polygons
        ax.set_xlim(left=0, right=img.shape[0])
        ax.set_ylim(bottom=img.shape[1], top=0)

    xs, ys = df["x_fit"], df["y_fit"]
    vor = spatial.Voronoi([(x, y) for x, y in zip(xs, ys)])

    cluster_members = _moran_cluster_members(df, sig)
    # Colorize the voronoi cells
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            axs[0].fill(*zip(*polygon),
                        color=mapper.to_rgba(df.iloc[r]["dot_normal_disp"]), alpha=0.5)
            if r in cluster_members:
                if df.iloc[r]["moran_quad"] in {1, 4}:
                    clust_kind = to_rgba("goldenrod", alpha=0.5)
                elif df.iloc[r]["moran_quad"] in {2, 3}:
                    clust_kind = to_rgba("maroon", alpha=0.5)
                else:
                    clust_kind = to_rgba("black", alpha=0.5)
                axs[1].fill(*zip(*polygon), c=clust_kind)
            else:
                axs[1].fill(*zip(*polygon), c=(0.1, 0.05, 0.05, 0.5))
    for ax in axs:
        ax.quiver(xs, ys, df["disp"].str[0], df["disp"].str[1], color="#C0C0C0")


def _geary_cluster_members(df, sig):
    core_members = {i for i, (p, c) in enumerate(zip(df["geary_p"], df["geary_C"]))
                    if (p <= sig) & (c <= 2)}  # 2 is the expected value of Ci
    members = set(core_members)
    for m in core_members:
        members.update(df.iloc[m]["neighborhood"])
    return members


def plot_geary_clusters(df, img, sig=0.05, var_cmap=plt.get_cmap("cmr.amber"), image_cmap="bone"):
    # plt.style.use("dark_background")  # For testing, because it's easier to look at on a screen
    plt.style.use("seaborn-v0_8-colorblind")
    # Mappers for colorizing the voronoi plots
    norm = Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=var_cmap)

    fig, axs = plt.subplots(1, 3, width_ratios=[1.068, 1, 1])
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0)

    axs[0].set_title("Normalized Intensity")
    fig.colorbar(mapper, ax=axs[0], location="left", fraction=0.04, pad=0.02)
    axs[1].set_title("Normalized Locally Correlated Displacement")
    axs[2].set_title("Multivariate Clusters")

    # Each set of axes gets an image
    for ax in axs:
        ax.imshow(img, cmap=image_cmap)
        ax.axis("off")
        # Set limits to keep them from being distorted by the voronoi polygons
        ax.set_xlim(left=0, right=img.shape[0])
        ax.set_ylim(bottom=img.shape[1], top=0)

    cluster_members = _geary_cluster_members(df, sig=sig)
    int_upper, int_lower = (np.mean(df["norm_int"]), np.mean(df["norm_int"]))
    disp_upper, disp_lower = (np.mean(df["dot_normal_disp"]), np.mean(df["dot_normal_disp"]))

    xs, ys = df["x_fit"], df["y_fit"]
    vor = spatial.Voronoi([(x, y) for x, y in zip(xs, ys)])

    # Colorize the voronoi cells
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            axs[0].fill(*zip(*polygon),
                        color=mapper.to_rgba(df.iloc[r]["norm_int"]), alpha=0.5)
            axs[1].fill(*zip(*polygon),
                        color=mapper.to_rgba(df.iloc[r]["dot_normal_disp"]), alpha=0.5)
            
            if r in cluster_members:
                if df.iloc[r]["dot_normal_disp"] >= disp_upper:
                    clust_kind = (*to_rgb("goldenrod"), 0.5)
                elif df.iloc[r]["dot_normal_disp"] < disp_lower:
                    clust_kind = (*to_rgb("maroon"), 0.5)
            else:
                clust_kind = (0, 0, 0, 0.5)
            axs[2].fill(*zip(*polygon), color=clust_kind)

    # Add arrows to the displacement plot
    axs[1].quiver(xs, ys, df["disp"].str[0], df["disp"].str[1], color="#C0C0C0")
    axs[2].quiver(xs, ys, df["disp"].str[0], df["disp"].str[1], color="#C0C0C0")

    # Legend for the cluster plot
    legend_elements = [Line2D([0], [0], marker="o", linestyle="none", markeredgecolor="none",
                              markerfacecolor="goldenrod",
                              label="Polar Cluster"),
                       Line2D([0], [0], marker="o", linestyle="none", markeredgecolor="none",
                              markerfacecolor="maroon",
                              label="Non-Polar Cluster")]
    axs[2].legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.20), loc="lower center")

    plt.show()


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
