# %% IMPORTS

# ASE Imports
from ase import Atoms
import ase.build as abuild
# AbTEM Imports
from abtem.potentials import Potential
from abtem.structures import orthogonalize_cell
from abtem.temperature import FrozenPhonons
from abtem.transfer import CTF
from abtem.waves import Probe, SMatrix
from abtem.scan import GridScan
from abtem.detect import FlexibleAnnularDetector
# Other Imports
from typing import Dict
from copy import deepcopy
from random import seed, choices, randint
from sys import maxsize
from scipy import spatial, stats
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgba
import esda
from libpysal import weights
import cupy
import os
import pickle
import SingleOrigin as so
import numpy as np
import matplotlib.pyplot as plt

# %% UTIL FUNCTIONS


def randomize_chem(atoms: Atoms,
                   replacements: Dict[str, Dict[str, float]],
                   prseed: int = randint(0, maxsize)) -> Atoms:
    """Randomize the chemistry of an ASE ``Atoms`` object via to user-defined replacement rules.

    Parameters
    ----------
    atoms : Atoms
        Initial ASE ``Atoms`` object.  A changed copy of this object will be returned.
    replacements : Dict[str, Dict[str, float]]
        Replacement dictionary.  The keys should be the symbols of the initial elements to replace,
        and the values should themselves be dictionaries.  The value dicts should have keys which
        are the elements that will replace the corresponding initial element, and the the values
        should be floats representing the fraction of the initial element to replace with the given
        element.  The sum of the floats must be <= 1 for each initial element.  For example:
            >>> {"Ba": {"Sr": 1},
            >>>  "Ti": {"Zr": 0.4,
            >>>         "Nb": 0.05}}
        would replace all Ba atoms in ``atoms`` with Sr, would randomly replace 40% of Ti atoms in
        ``atoms`` with Zr, and randomly replace 5% (of the initial amount of Ti) with Nb.
    prseed : int, optional
        Pseudo-random seed.  The default is to randomly choose a seed between 0 and sys.maxsize.

    Returns
    -------
    Atoms
        ASE ``Atoms`` object based on ``atoms``, but with the specified elemental replacements.
    """
    seed(prseed)
    new_atoms = deepcopy(atoms)

    # Sanity check:
    for elem, rep in replacements.items():
        if sum(rep.values()) < 1:  # Add in the "NOP weights" (chance to not replace) if needed
            rep[elem] = 1 - sum(rep.values())
        assert sum(rep.values()) == 1  # If this is ever False, we're likely to get garbage results

    symbols = new_atoms.get_chemical_symbols()
    counts = dict(zip(set(symbols), [symbols.count(e) for e in set(symbols)]))

    for elem, reps in replacements.items():
        elem_idxs = [idx for idx, sym in enumerate(symbols) if sym == elem]
        rep_with = choices(list(reps.keys()), weights=list(reps.values()), k=counts[elem])
        for i in elem_idxs:
            symbols[i] = rep_with.pop()

    new_atoms.set_chemical_symbols(symbols)
    return new_atoms


# %% BUILD CELL

uc = Atoms("BaTiO3",
           cell=[4.083, 4.083, 4.083],
           pbc=True,
           scaled_positions=[(0, 0, 0),
                             (0.5, 0.5, 0.5),
                             (0.5, 0.5, 0),
                             (0.5, 0, 0.5),
                             (0, 0.5, 0.5)])
surface_001 = abuild.surface(uc, (0, 0, 1), layers=1, periodic=True)
orthogonalize_cell(surface_001)

big_cell = surface_001 * (12, 12, 12)
big_cell = randomize_chem(big_cell, {"Ti": {"Zr": 0.4}}, prseed=666)


cluster_coords = [(x, y, z) for x in range(4, 8) for y in range(4, 8) for z in range(7, 11)]

# Magic numbers are SPECIFIC TO THIS 12x12 MODEL, and will need adjusting if the model is scaled
cluster_idxs = [1 + 720*x + 60*y + 5*z for x, y, z in cluster_coords]

symbols = big_cell.get_chemical_symbols()
symbols = ["Zr" if i in cluster_idxs else symbols[i] for i in range(len(symbols))]
big_cell.set_chemical_symbols(symbols)

# %% PROJECTED COUNT MAP
count_map = np.ndarray((12, 12))
for x in range(12):
    for y in range(12):
        count = 0
        for z in range(12):
            current_idx = 1 + 720*x + 60*y + 5*z
            if big_cell.numbers[current_idx] == 40:
                count += 1
        count_map[x][y] = count

# %% SETUP FOR SIM

# Setup potential with frozen phonons
phonon_model = FrozenPhonons(big_cell,
                             sigmas={"Ba": 0.0757,
                                     "Ti": 0.0893,
                                     "Zr": 0.1050,
                                     "O":  0.0810},
                             num_configs=100)
potential = Potential(phonon_model,
                      sampling=0.04,
                      device="gpu",
                      storage="gpu",
                      projection="infinite",
                      parametrization="kirkland",
                      slice_thickness=1)

# Setup probe, scan grid, and detector
probe = Probe(energy=200E3,
              semiangle_cutoff=17.9,
              device="gpu")
probe.ctf.set_parameters({"astigmatism": 7,
                          "astigmatism_angle": 155,
                          "coma": 300,
                          "coma_angle": 155,
                          "Cs": -5000})
# prisim_ctf = CTF(semiangle_cutoff=17.9,
#                  energy=200E3,
#                  parameters={"astigmatism": 7,
#                              "astigmatism_angle": 155,
#                              "coma": 300,
#                              "coma_angle": 155,
#                              "Cs": -5000})
# probe = SMatrix(energy=200E3,
#                 expansion_cutoff=20,
#                 interpoalition=4,
#                 ctf=prisim_ctf)
scan = GridScan(start=[0, 0], end=potential.extent, sampling=probe.ctf.nyquist_sampling)
detector = FlexibleAnnularDetector()

# %% PROPAGATE

with cupy.cuda.Device(0):
    measurements = []
    probe.grid.match(potential)
    measurement = probe.scan(scan, [detector], potential, pbar=True)
    measurements.append(measurement)

# %% PICKLE MEASUREMENT
export_path = r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_meas_pickle"
if not os.path.exists(export_path):
    os.makedirs(export_path)

with open(os.path.join(export_path, "big_cluster_test_cube_seed666_empty"), "wb+") as outfile:
    pickle.dump(measurements, outfile)

# %% UNPICKLE
import_path = r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_meas_pickle"
with open(os.path.join(import_path, "big_cluster_test_cube_seed666_empty"), "rb") as infile:
    measurement = pickle.load(infile)[0]

# %% DETECT
meas = deepcopy(measurement)  # Don't mutate original measurements, or must redo sim
haadf = meas.integrate(70, 200)  # Integration limits from camera length
# Interpolate using 4xFFT
# Demonstrated by abTEM author to be nearly identical to 4x oversampling, much faster
haadf = haadf.interpolate(tuple([x/4 for x in scan.sampling]), kind="fft")

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
# uc.plot_unit_cell()  # Check this output to make sure it's sensible

# %% CREATE HRIMAGE OBJECT
hr_img = so.HRImage(haadf.array)
lattice = hr_img.add_lattice("BZT", uc)

# %% BASIS VECTOR DEFINITION

# Get real space basis vectors using the FFT
# If some FFT peaks are weak or absent (such as forbidden reflections),
#  specify the order of the first peak that is clearly visible
lattice.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)

# %% DEFINE REGION MASK
lattice.region_mask = np.ones(haadf.array.shape)

# %% REFERENCE LATTICE FIT
lattice.define_reference_lattice()

# %% ATOM COLUMN FITTING

# Fit atom columns at reference lattice points
lattice.fit_atom_columns(buffer=1, local_thresh_factor=0.5,
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

# %% AUGMENT DATAFRAME

# A copy of the at_cols dataframe of the lattice object, for convenience
bframe = deepcopy(lattice.at_cols)


def _fmt_neighbors(row, df, kdtree):
    """Helper function to reformat neighbor array to use the correct indices."""
    _, idxs = row["neighbors"]
    no_nn = max(idxs)  # The kdtree querry uses this value as a standin for "no more neighbors"
    idxs = list(filter(lambda x: x != no_nn, idxs))
    new_idxs = []
    for i in idxs:
        # Relocate the correct index in the dataframe reference (the tree only indexes A sites)
        x, y = kdtree.data[i]
        match = df.loc[df["x_fit"] == x]
        if len(match) != 1:
            raise RuntimeError("Something went wrong!")
        new_idxs.append(match.index[0])
    # Sometimes we find no neighbors if the column was fit poorly; just throw it out
    if new_idxs == []:
        new_idxs = None
    return new_idxs


def _get_int_ratios(row, df):
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


def compute_neighborhood_stats(df, maxdist=55, n=4, a_elem="Ba", outlier_scale=10):
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
    df["int_ratio"] = df.apply(lambda row: _get_int_ratios(row, df), axis=1)
    df["int_ratio"] = df.apply(lambda row: _reject_outliers(row, df,
                                                            outlier_scale), axis=1)
    # Compute the dispersion
    df["disp"] = df.apply(lambda row: (row["x_fit"] - row["x_ref"],
                                       row["y_fit"] - row["y_ref"]), axis=1)

    # Discard the A sites, and any outliers, then drop irrelevant columns
    drop_idxs = df[np.isnan(df["int_ratio"])].index
    df.drop(drop_idxs, inplace=True)
    df.drop(["elem", "site_frac", "x", "y", "weight", "neighbors"],
            axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)


compute_neighborhood_stats(bframe, maxdist=90)

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

ratios = [r for r in bframe["int_ratio"]]

mor = esda.moran.Moran(ratios, w, permutations=10000)
print(f"Global Moran's I:     {round(mor.I, 5)}\n"
      f"Expected I for CSR:  {round(mor.EI_sim, 5)}\n"
      f"p-value (10k perms.): {round(mor.p_sim, 5)}")

# %% PYSAL ANALYSIS -- LOCAL MORAN'S I

norm = Normalize(vmin=min(bframe["int_ratio"]),
                 vmax=max(bframe["int_ratio"]), clip=True)
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
    alpha_arr = [(1,) if p <= max(p_bands) and (q != 2) and (q != 4)
                 else (0,) for p, q in zip(ps, quads)]

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
axs[0].imshow(haadf.array, cmap="gray")
axs[1].imshow(haadf.array, cmap="gray")

xs, ys = bframe["x_fit"], bframe["y_fit"]
s = 70
axs[0].scatter(xs, ys, color=mapper.to_rgba(ratios), s=s, linewidths=0)

ps = [p for p in bframe["moran_p"]]  # REMEMBER: These are *pseudo* p-values
quads = [q for q in bframe["moran_quadrant"]]
axs[1].scatter(xs, ys, color=colorize_cluster(ps, quads), s=s, linewidths=0)

axs[0].axis("off")
axs[1].axis("off")
fig.colorbar(mapper, ax=axs[0], label="B/A Intensity Ratio", location="left", fraction=0.04)
