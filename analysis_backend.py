from collections.abc import Sequence
from copy import deepcopy
from typing import Literal
from esda import Moran, Moran_Local, Moran_BV, Moran_Local_BV, Geary, Geary_Local, Geary_Local_MV
from libpysal import weights
from libpysal.weights import W
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, Colormap, to_rgba, to_rgb
from matplotlib.lines import Line2D
from numpy import ndarray, outer, asarray, isclose, NaN, median, isnan, array, sum, dot, mean, std
from numpy.linalg import linalg
from pandas import DataFrame, Series
from scipy import signal, ndimage
from scipy.spatial import KDTree, Voronoi
from scipy.stats import median_abs_deviation


def divide_image_frequencies(img: ndarray[float],
                             s: float | Sequence[float],
                             show_gauss: bool = False,
                             show_images: bool = False)\
        -> tuple[ndarray[float, ...], ndarray[float, ...]]:
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
        g2d = outer(g1d, g1d)
        plt.imshow(g2d, cmap="magma")

    lowpass = ndimage.gaussian_filter(img, sigma=s, mode="reflect")
    highpass = img - lowpass

    if show_images:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(lowpass, cmap="magma")
        axs[1].imshow(highpass, cmap="magma")
    return lowpass, highpass


def grow_tree(df: DataFrame,
              a2d: ndarray,
              kind: str | list[str] | None)\
        -> KDTree:
    """
    Parameters
    ----------
    df : DataFrame
        The dataframe containing the columns relevant to growing the 2D near neighbor tree
        ("elem", "u", and "v")
    a2d : ndarray
        2x2 ndarray to convert from pixel coordinates to coordinates in A
    kind : String or list of strings or None
        String(s) representing the kind of atom columns which are valid neighbors; matches against
        the "elem" column in df.  A value of None matches all kinds of columns

    Returns
    -------
    numpy kdtree (2D) that can be used for fast near-neighbor searching.
    """
    fractional_coords = df.loc[df["elem"] == kind, "u":"v"].to_numpy()
    coords_in_A = fractional_coords @ a2d
    tree = KDTree(coords_in_A)
    return tree


def get_near_neighbors(row: Series | DataFrame,
                       df: DataFrame,
                       a2d: ndarray,
                       tree: KDTree,
                       n: int)\
        -> list[int] | None:
    """
    Parameters
    ----------
    row : DataFrame row
        The row representing the site that's currently being searched around
    df : DataFrame
        The DataFrame represening the sites fitted in the image
    a2d : ndarray
        2x2 ndarray to convert from pixel coordinates to coordinates in A
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
    site = asarray((u, v)) @ a2d
    near_neighbors = tree.query(site, k=n+1)

    # The kdtree query returns an unwieldy format for the neighborhood, which we need to fix
    no_nn = tree.n  # This k-D tree implementation uses self.n to indicate a missing neighbor
    idxs = list(filter(lambda idx: idx != no_nn, near_neighbors[1]))
    neighborhood = []
    for i in idxs:
        x, y = tree.data[i]
        a2d_inv = linalg.inv(a2d)
        u_match, v_match = asarray((x, y)) @ a2d_inv
        match = df.loc[(isclose(df["u"], u_match)) & (isclose(df["v"], v_match))]
        if len(match) != 1:
            raise RuntimeError("No near-neighbor (u, v) matches found; this shouldn't happen!")
        neighborhood.append(match.index[0])
    # Sometimes we find no neighbors if the column was fit poorly; just throw it out
    if not neighborhood:
        neighborhood = None
    return neighborhood


def _test_outliers(x: float,
                   med: ndarray,
                   mad: float,
                   outlier_scale: float)\
        -> float | NaN:
    """Returns NaN if x is an outlier, else returns x"""
    if abs(x - med) > outlier_scale*mad:
        return NaN
    else:
        return x


def reject_outliers(df: DataFrame,
                    mode: Literal["total_col_int", "disp_mag"],
                    outlier_scale: float = 10)\
        -> None:
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
    rmed = median(df[mode])
    rmad = median_abs_deviation(df[mode])

    df[mode] = df.apply(lambda row: _test_outliers(row[mode], rmed, rmad, outlier_scale), axis=1)

    outlier_count = df[mode].isna().sum()
    if outlier_count != 0:
        print(f"Dropping {outlier_count} outlier site{'s'[:outlier_count^1]}" +
              f" based on values in '{mode}'.")
        df.drop(df.loc[isnan(df[mode])].index, inplace=True)
        df.reset_index(drop=True, inplace=True)


def _dot_disp(row: Series | DataFrame,
              df: DataFrame)\
        -> ndarray:
    neighbor_disps = list(map(array, [df.iloc[i]["disp"] for i in row["neighborhood"]]))
    mean_neighbor_disp = sum(neighbor_disps, axis=0)/len(neighbor_disps)
    return dot(row["disp"], mean_neighbor_disp)


def disp_calc(df: DataFrame,
              kind: str | None = None,
              normalize: bool = False)\
        -> None:
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
        Whether to add a column with (globally) normalized values for `dot_disp`.  The
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


def drop_elements(df: DataFrame,
                  e: list[str])\
        -> None:
    """
    Parameters
    ----------
    df : DataFrame
        The DataFrame represening the sites fitted in the image.
        Warning: this function *will mutate* df
    e : list of string
        The elements to be removed from df, matched against the "elem" column

    Returns
    -------
    None, but mutates df by removing rows that match any of the elements in elem, and then
    resetting the indices
    """
    df.drop(df.loc[df["elem"].isin(e)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _global_minmax(row: Series | DataFrame,
                   df: DataFrame,
                   kind: str | None)\
        -> float:
    """Normalization mode for column intensity"""
    site_intensity = row["total_col_int"]
    if kind is None:
        min_int, max_int = min(df["total_col_int"]), max(df["total_col_int"])
    else:
        min_int = min(df.loc[df["elem"] == kind, "total_col_int"])
        max_int = max(df.loc[df["elem"] == kind, "total_col_int"])
    return (site_intensity - min_int) / (max_int - min_int)


def _ratio(row: Series | DataFrame,
           df: DataFrame)\
        -> float:
    """Normalization mode for site intensity"""
    site_intensity = row["total_col_int"]
    neighbor_intensities = [df.iloc[i]["total_col_int"] for i in row["neighborhood"]]
    return site_intensity / mean(neighbor_intensities)


def _standard_score(row: Series | DataFrame,
                    df: DataFrame)\
        -> float:
    """Normalization mode for site intensity"""
    site_intensity = row["total_col_int"]
    neighbor_intensities = [df.iloc[i]["total_col_int"] for i in row["neighborhood"]]
    avg, stdev = mean(neighbor_intensities), std(neighbor_intensities)
    return (site_intensity - avg) / stdev


def _off(row: Series | DataFrame)\
        -> float:
    """Normalization mode for site intensity (doesn't normalize, just returns)"""
    return row["total_col_int"]


def normalize_intensity(df: DataFrame,
                        a2d: ndarray,
                        n: int = 4,
                        method: Literal["Ratio", "standard_score", "off", "global_minmax"] | None = "ratio",
                        kind: str | None = None)\
        -> None:
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


def _rookzone(df: DataFrame,
              row: Series | DataFrame)\
        -> list[int]:
    """Rook neighborhood zone shape:
        [(u-1, v), (u+1, v), (u, v+1), (u, v-1)]"""
    u_m = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"])]
    u_p = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"])]
    v_m = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] - 1)]
    v_p = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] + 1)]

    return [n.index[0] for n in [u_m, u_p, v_m, v_p] if n.size > 0]


def _bishopzone(df: DataFrame,
                row: Series | DataFrame)\
        -> list[int]:
    """Bishop neighborhood zone shape:
        [(u+1, v+1), (u-1, v-1), (u+1, v-1), (u-1, v+1)]"""
    uv_pp = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"] + 1)]
    uv_mm = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"] - 1)]
    uv_pm = df.loc[(df["u"] == row["u"] + 1) & (df["v"] == row["v"] - 1)]
    uv_mp = df.loc[(df["u"] == row["u"] - 1) & (df["v"] == row["v"] + 1)]

    return [n.index[0] for n in [uv_pp, uv_mm, uv_pm, uv_mp] if n.size > 0]


def _120zone(df: DataFrame,
             row: Series | DataFrame,
             a2d: ndarray)\
        -> list[int]:
    """Sepecial neighborhood zone shape for perovskite 120 zone axis:
        [(shorter(u,v)+2, longer(u,v)), (shorter(u,v)-2, longer(u,v))]"""
    # Need to get 2nd NNs in the B row ==> extra neighbors on shorter of u or v
    if linalg.norm(a2d[0]) < linalg.norm(a2d[1]):  # u is shorter
        nn2_up = df.loc[(df["u"] == row["u"] + 2) & (df["v"] == row["v"])]
        nn2_dn = df.loc[(df["u"] == row["u"] - 2) & (df["v"] == row["v"])]
    else:  # v is shorter
        nn2_up = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] + 2)]
        nn2_dn = df.loc[(df["u"] == row["u"]) & (df["v"] == row["v"] - 2)]

    return [n.index[0] for n in [nn2_up, nn2_dn] if n.size > 0]


def make_weights(df: DataFrame,
                 adj_type: Literal["rook", "queen", "king", "bishop", "120"],
                 a2d: ndarray | None = None)\
        -> W:
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
    w : libpysal W (weights) object
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


def _sink_islands(df: DataFrame,
                  w: W):
    """
    Parameters
    ----------
    df : DataFrame
        The frame containing sites which are islands.
        Warning: this will mutate the frame by dropping island rows!
    w : libpysal.weights.W
        Weights object defining adjacency: this is the object that knows which sites are islands
    -------
    w : libpysal.weights.W
        Weights object with the islands removed

    """
    islands = frozenset(w.islands)  # Having this be a set makes everything simpler and faster
    print(f"Dropping {len(islands)} outlier site{'s'[:len(islands)^1]} based on 'w.islands'")
    w = weights.w_subset(w, list(set(w.id_order) - islands))

    df.drop(index=islands, inplace=True)
    return w


def get_stats(df: DataFrame,
              adj_type: Literal["rook", "queen", "king", "bishop", "120"],
              a2d: ndarray,
              kind: Literal["moran_global", "moran_local", "moran_global_bivariate", "moran_local_bivariate",
                            "geary_global", "geary_local", "geary_local_multivariate"],
              columns: str | Sequence[str],
              p: int = 10000,
              printstats: bool = True)\
        -> Moran | Moran_Local | Moran_BV | Moran_Local_BV | Geary | Geary_Local | Geary_Local_MV:
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
        Whether to print some statistics to the console. This setting only applies to global
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
            # Valid columns types: string or sequence containing exactly one string
            if not (isinstance(columns, str) or (isinstance(columns, Sequence)
                                                 and len(columns) == 1)):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")
            if not isinstance(columns, str):
                columns = columns[0]  # If the string is in another sequence, unwrap it

            vals = [val for val in df[columns]]

            if kind == "moran_global":
                # noinspection PyTypeChecker
                stat_blob = Moran(vals, w, permutations=p)
                if printstats:
                    print(f"Global Moran's I:     {round(stat_blob.I, 5)}\n"
                          f"Expected I for CSR:  {round(stat_blob.EI_sim, 5)}\n"
                          f"p-value ({p} perms.): {round(stat_blob.p_sim, 5)}")
                    if stat_blob.p_sim < 0.05:
                        print("Observed distribution not consistent with CSR")
                    else:
                        print("Observed distribution consistent with CSR")

            elif kind == "moran_local":
                # noinspection PyTypeChecker
                stat_blob = Moran_Local(vals, w, permutations=p)

        case "moran_global_bivariate" | "moran_local_bivariate":
            # Valid columns types: iterable containing exactly two strings
            if not (isinstance(columns, Sequence) and len(columns) == 2):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")

            vals = [[val for val in df[col]] for col in columns]

            if kind == "moran_global_bivariate":
                # noinspection PyTypeChecker
                stat_blob = Moran_BV(vals[0], vals[1], w, permutations=p)
                if printstats:
                    print(f"Global Moran's I:     {round(stat_blob.I, 5)}\n"
                          f"Expected I for CSR:  {round(stat_blob.EI_sim, 5)}\n"
                          f"p-value ({p} perms.): {round(stat_blob.p_sim, 5)}")
                    if stat_blob.p_sim < 0.05:
                        print("Observed distribution not consistent with CSR")
                    else:
                        print("Observed distribution consistent with CSR")

            elif kind == "moran_local_bivariate":
                # noinspection PyTypeChecker
                stat_blob = Moran_Local_BV(vals[0], vals[1], w, permutations=p)

        # ====== # GEARY STATISTICS # ====== #
        case "geary_global" | "geary_local":
            # Valid columns types: string or iterable containing exactly one string
            if not (isinstance(columns, str) or (isinstance(columns, Sequence)
                                                 and len(columns) == 1)):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")
            if not isinstance(columns, str):
                columns = columns[0]  # If the string is in another iterable, unwrap it

            vals = [val for val in df[columns]]

            if kind == "geary_global":
                # noinspection PyTypeChecker
                stat_blob = Geary(vals, w, permutations=p)
                if printstats:
                    print(f"Global Gear's C':     {round(stat_blob.C, 5)}\n"
                          f"Expected C for CSR:  {round(stat_blob.EC_sim, 5)}\n"
                          f"p-value ({p} perms.): {round(stat_blob.p_sim, 5)}")
                    if stat_blob.p_sim < 0.05:
                        print("Observed distribution not consistent with CSR")
                    else:
                        print("Observed distribution consistent with CSR")

            elif kind == "geary_local":
                stat_blob = Geary_Local(connectivity=w, labels=True, permutations=p)
                # noinspection PyTypeChecker
                stat_blob.fit(vals)

        case "geary_local_multivariate":
            if not (isinstance(columns, Sequence) and len(columns) >= 2):
                raise RuntimeError("Mismatch between `kind` and `columns` parameters")

            vals = [[val for val in df[col]] for col in columns]

            stat_blob = Geary_Local_MV(connectivity=w, permutations=p)
            # noinspection PyTypeChecker
            stat_blob.fit(vals)

        case _:
            raise NotImplementedError("The requested statistical test has not been implemented," +
                                      " or `kind` was an invalid string")
    # noinspection PyUnboundLocalVariable
    return stat_blob


def add_stats_to_frame(df: DataFrame,
                       sts: Moran | Moran_Local | Moran_BV | Moran_Local_BV | Geary | Geary_Local | Geary_Local_MV,
                       kind: Literal["moran", "geary"])\
        -> None:
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
    match kind:
        case "moran":
            df["moran_I"] = sts.Is
            df["moran_quad"] = sts.q
        case "geary":
            df["geary_C"] = sts.localG
            if "labs" in sts.__dict__:
                df["geary_label"] = sts.labs


def _moran_cluster_members(df: DataFrame,
                           sig: float):
    core_members = {i for i, p in enumerate(df["moran_p"]) if (p <= sig)}
    members = set(core_members)
    for m in core_members:
        members.update(df.iloc[m]["neighborhood"])
    return members


def plot_moran_clusters(df: DataFrame,
                        img: ndarray,
                        var_cmap: str | Colormap = plt.get_cmap("cmr.amber"),
                        image_cmap: str | Colormap = "bone",
                        sig: float = 0.05)\
        -> None:
    # plt.style.use("dark_background")  # For testing, because it's easier to look at on a screen
    plt.style.use("seaborn-v0_8-colorblind")
    # Mappers for colorizing the voronoi plots
    norm = Normalize(vmin=0, vmax=1, clip=True)
    mapper = ScalarMappable(norm=norm, cmap=var_cmap)

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
    # noinspection PyTypeChecker
    vor = Voronoi([(x, y) for x, y in zip(xs, ys)])

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


def _geary_cluster_members(df: DataFrame,
                           sig: float)\
        -> set[int]:
    core_members = {i for i, (p, c) in enumerate(zip(df["geary_p"], df["geary_C"]))
                    if (p <= sig) & (c <= 2)}  # 2 is the expected value of Ci
    members = deepcopy(core_members)
    for m in core_members:
        members.update(df.iloc[m]["neighborhood"])
    return members


def plot_geary_clusters(df: DataFrame,
                        img: ndarray,
                        sig: float = 0.05,
                        var_cmap: str | Colormap = plt.get_cmap("cmr.amber"),
                        image_cmap: str | Colormap = "bone")\
        -> None:
    # plt.style.use("dark_background")  # For testing, because it's easier to look at on a screen
    plt.style.use("seaborn-v0_8-colorblind")
    # Mappers for colorizing the voronoi plots
    norm = Normalize(vmin=0, vmax=1, clip=True)
    mapper = ScalarMappable(norm=norm, cmap=var_cmap)

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
    disp_upper, disp_lower = (mean(df["dot_normal_disp"]), mean(df["dot_normal_disp"]))

    xs, ys = df["x_fit"], df["y_fit"]
    # noinspection PyTypeChecker
    vor = Voronoi([(x, y) for x, y in zip(xs, ys)])

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
            # noinspection PyUnboundLocalVariable
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
