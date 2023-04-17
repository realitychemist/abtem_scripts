# %% IMPORTS

# ASE Imports
from ase import Atoms
import ase.build as abuild
# AbTEM Imports
from abtem.potentials import Potential
from abtem.structures import orthogonalize_cell
from abtem.temperature import FrozenPhonons
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.detect import FlexibleAnnularDetector
# Other Imports
import os
import pickle
import cupy
import SingleOrigin as so
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from labellines import labelLine
from tqdm import tqdm
from copy import deepcopy
from random import seed, choices
from typing import Dict
from ase_funcs import randomize_chem
import abtem_backend as backend

# %% SETUP
uc = Atoms("BaTiO3",
           cell=[4.083, 4.083, 4.083],
           pbc=True,
           scaled_positions=[(0, 0, 0),
                             (0.5, 0.5, 0.5),
                             (0.5, 0.5, 0),
                             (0.5, 0, 0.5),
                             (0, 0.5, 0.5)])
atoms = abuild.surface(uc, (0, 0, 1), layers=1, periodic=True)
atoms = orthogonalize_cell(atoms)

stack = atoms * (1, 1, 36)  # 36 --> 14.7nm

# Random seeds from random.org
seeds = [87353, 39801, 56916, 62903, 76446, 40231, 92312, 43299, 72148, 37976,
         93458, 22838, 78787, 89538, 53240, 82349, 85799, 94281, 53053, 10655,
         94124, 6828, 21401, 75500, 7576, 74045, 70885, 23437, 25341, 59347,
         60517, 66924, 78696, 48347, 580, 51813, 42469, 29790, 59860, 53418,
         89435, 44210, 62350, 82493, 92909, 64157, 4272, 86548, 78072, 33308,
         44844, 59068, 71774, 9102, 15659, 15109, 51366, 28656, 53572, 81414]

randomized_atoms = [randomize_chem(stack, {"Ti": {"Zr": 0.3}}, prseed=s) for s in seeds]
randomized_atoms = [m * (2, 2, 1) for m in randomized_atoms]  # 2x2 needed to avoid probe wraparound

# Frozen phonons for each model
# No reason not to reuse the same seeds to generate the frozen phonons
models = [_gen_phonons(model, seeds[i]) for i, model in enumerate(models)]
# Prebuild potentials
potentials = [_gen_potentials(model) for model in models]

# Setup probe, scan grid, and detector
probe = Probe(energy=200E3,
              semiangle_cutoff=17.9,
              device="gpu")
probe.ctf.set_parameters({"astigmatism": 7,
                          "astigmatism_angle": 155,
                          "coma": 300,
                          "coma_angle": 155,
                          "Cs": -5000})
scan = GridScan(start=[0, 0], end=[4.083, 4.083], sampling=probe.ctf.nyquist_sampling*0.9)
# FA detector will let us easily test different camera lengths without redoing the sim
detector = FlexibleAnnularDetector()


# %% PROPAGATE

with cupy.cuda.Device(1):
    measurements = []
    for potential in potentials:
        probe.grid.match(potential)  # Could technically happen outside loop, but it's fast & works
        measurement = probe.scan(scan, [detector], potential, pbar=False)
        measurements.append(measurement)

# %% PICKLE MEASUREMENTS
export_path = r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_meas_pickle"
if not os.path.exists(export_path):
    os.makedirs(export_path)

with open(os.path.join(export_path, "meas_30"), "wb+") as outfile:
    pickle.dump(measurements, outfile)

# %% DETECT
# Measurement in a separate loop to easily try different CL w/o redoing sim
haadfs = []
for measurement in measurements:
    meas = deepcopy(measurement)  # Don't mutate original measurements, or must redo sim

    haadf = meas.integrate(70, 200)  # Integration limits from camera length
    # Interpolate using 4xFFT
    # Demonstrated by abTEM author tobe nearly identical to 4x oversampling, much faster
    haadf = haadf.interpolate(tuple([x/4 for x in scan.sampling]), kind="fft").tile((3, 3))
    haadfs.append(haadf)

# %% SINGLEORIGIN SETUP
# Setup SingleOrigin
# The unit cell in this cif file is tetragonal, but that shouldn't matter
CIF_PATH = r"E:\Users\Charles\BaTiO3 Controls\No Shift\BaTiO3_99_noshift.cif"

za = [0, 0, 1]  # Zone axis direction
a2 = [1, 0, 0]  # Apparent horizontal axis in projection
a3 = [0, 1, 0]  # Most vertical axis in projection

# Initialize UnitCell object
uc = so.UnitCell(CIF_PATH)
uc.transform_basis(za, a2, a3)
# Ignore light elements for HAADF
uc.project_uc_2d(proj_axis=0, ignore_elements=["O"])
uc.combine_prox_cols()

# Normalize images for use with SingleOrigin
normed_haadf = [so.image_norm(haadf.array) for haadf in haadfs]

# Build SO AtomicColumnLattice object, only need to build once
acl = so.AtomicColumnLattice(normed_haadf[0], uc, probe_fwhm=0.76, origin_atom_column=0)
# Hack the atomic column lattice objects to avoid issues with the low-res FFT
uc_pixels = int(normed_haadf[0].shape[0]/3)
acl.dir_struct_matrix = np.array([[uc_pixels, 0],
                                  [0, uc_pixels]])
acl.a1, acl.a2 = acl.dir_struct_matrix[0], acl.dir_struct_matrix[1]
acl.a_2d = np.array([[4.083, 0],
                     [0, 4.083]])
acl.pixel_size = 4.083 / uc_pixels
acl.define_reference_lattice()  # Interactive step: pick top left most full A column


# %% SINGLEORIGIN FIND PEAKS

intensity_pairs = []
for img in normed_haadf:
    acl.image = img
    acl.fit_atom_columns(buffer=0, local_thresh_factor=3,
                         grouping_filter=None, diff_filter="auto", parallelize=False)
    a_intensity = acl.at_cols.iloc[9].total_col_int
    b_intensity = acl.at_cols.iloc[10].total_col_int
    intensity_pairs.append((a_intensity, b_intensity))

# %% RATIO STATISTICS

ratios = [b / a for a, b in intensity_pairs]
n, minmax, mean, var, skew, kurt = stats.describe(ratios)

stats.probplot(ratios, plot=plt)  # Save this by hand for each probe

# %% EXPORT DATA
export_path = r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_stats_pickle"
if not os.path.exists(export_path):
    os.makedirs(export_path)

export_bundle = {"label":    "BZT_0",  # MAKE SURE TO KEEP THIS UP TO DATE
                 "raw":      ratios,
                 "n":        n,
                 "minimum":  minmax[0],
                 "maximum":  minmax[1],
                 "mean":     mean,
                 "varience": var,
                 "st_dev":   np.sqrt(var),
                 "skewness": skew,
                 "kurtosis": kurt}

with open(os.path.join(export_path, export_bundle["label"]), "wb+") as outfile:
    pickle.dump(export_bundle, outfile)


# %% READ BACK STATS

with open(r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_stats_pickle\BZT_30", "rb") as infile:
    import_30 = pickle.load(infile)
with open(r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_stats_pickle\BZT_40", "rb") as infile:
    import_40 = pickle.load(infile)
with open(r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_stats_pickle\BZT_50", "rb") as infile:
    import_50 = pickle.load(infile)

imported_stats = [import_30, import_40, import_50]

# %% SHAPIRO-WILK NORMALITY TEST

sw_results = [stats.shapiro(s["raw"]) for s in imported_stats]
for label, result in zip(["BZT 30", "BZT 40", "BZT 50"], sw_results):
    print(f"{label} -- Statistic: {result[0]} -- p-Value: {result[1]}")

# %% PLOT INFERRED NORMAL DISTRIBUTIONS

colors = ["#D81B60", "#1E88E5", "#004D40"]  # IBM colorblind friendly colors

for i, (s, c) in enumerate(zip(imported_stats, colors)):
    start = s["mean"] - 3*s["st_dev"]
    stop = s["mean"] + 3*s["st_dev"]
    xs = np.linspace(start, stop, 1000)
    n = stats.norm.pdf(xs, s["mean"], s["st_dev"])
    plt.plot(xs, n, label=s["label"].replace("_", ""), color=c)

    ax = plt.gca()
    labelLine(ax.get_lines()[i], s["mean"]+s["st_dev"],
              label=s["label"].replace("_", ""), fontsize=12)

    minus_2s = s["mean"]-2*s["st_dev"]
    plus_2s = s["mean"]+2*s["st_dev"]
    yval = np.interp(minus_2s, xs, n)

    plt.vlines([minus_2s, plus_2s], 0, yval, linestyle="dashed", color=c)

plt.ylim((0, 8))
plt.tick_params(axis="y", which="both", left=False, labelleft=False)
# plt.title("B/A Intensity Ratio Distributions", fontsize=24)
plt.xlabel("B/A Intensity Ratio", fontsize=24)
plt.show()
