"""
THIS SCRIPT IS CURRENTLY NOT BEING USED, IT WAS TOO SLOW OF AN APPROACH
I'VE BEGUN DEVELOPING A SCRIPT WHICH ATTEMPTS TO DO WHAT THIS DOES BUT BY PICKING RANDOM
ARRANGEMENTS FOR A SUBSET OF POSSIBLE THICKNESSES
I'M KEEPING THIS BECAUSE I EXPECT IT TO BE USEFUL AT SOME POINT IN THE FUTURE
"""

# %% IMPORTS

# ASE Imports
from ase import Atoms
import ase.build as abuild
# AbTEM Imports
from abtem.potentials import Potential
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.detect import AnnularDetector, SegmentedDetector
from abtem.temperature import FrozenPhonons
# Other Imports
import os
import tifffile
import rle
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations_with_replacement, permutations
from random import seed, choices
from typing import Dict, Tuple

# %% SETTINGS
prms = {"seed":            42,             # Pseudo-random seed
        "export_path":     r"E:\Users\Charles\BZT Intensity Mapping",  # Export path, sans subdir
        # POTENTIAL SETTINGS
        "lattice_a":       4.083,          # Pseudo-cubic lattice parameter (A)
        "sampling":        0.04,           # Sampling of potential (A)
        "thickness":       100,            # Total model thickness (A);
                                           #  will be rounded up to a full projected cell
        "slice_thickness": 1,              # Thickness per slice (A)
        "zas":            [(0, 0, 1),      # Zone axis to model and simulate,
                           (1, 1, 0)],
        "fp_cfgs":         10,             # Number of frozen phonon configurations to simulate
        "fp_sigmas":      {"Ba": 0.0757,   # Frozen phonon sigma values per atom type
                           "Ti": 0.0893,   # sigma == sqrt(U_iso) (confirmed with abTEM author)
                           "Zr": 0.1050,   # Data from https://materials.springer.com/isp/crystallographic/docs/sd_1410590
                           "O":  0.0810},  # and from https://pubs.acs.org/doi/10.1021/acs.chemmater.9b04437 (for Zr)
        # PROBE SETTINGS
        "beam_energy":     200E3,          # Energy of the electron beam (eV)
        "conv_angle":      17.9,           # Probe semiangle of convergence (mrad)
        "defocus":         0,              # Defocus (A)
        "stig":            0,              # 1st-order (2-fold) astigmatism (A)
        "stig_rot":        0,              # A1 rotation (rad)
        "coma":            0,              # 2nd-order axial coma (A)
        "coma_rot":        0,              # B2 rotation (rad)
        "Cs":              0,              # Spherical aberration (A)
        "max_batch":       50,             # Number of probe positions to propogate at once
        # DETECTOR SETTINGS
        "haadf_min_angle": 69,             # HAADF annulus inner extent (mrad)
        "haadf_max_angle": 200,            # HAADF annulus outer extent (mrad)
        "df4_min_angle":   16,             # DF4 annulus inner extent (mrad)
        "df4_max_angle":   65,             # DF4 annulus outer extent (mrad)
        "df4_rotation":    19*np.pi / 5}   # Rotation of DF segments (rad)

# %% CUSTOM BUILD FUNCTIONS


def gen_arrangements(model: Atoms, bs: Tuple[str, str]) -> Tuple[str, Atoms]:
    """Generate ASE ``Atoms`` models by all possible combinations and permustations of the B site.

    Parameters
    ----------
    model : Atoms
        The model to permute the chemistry of.  Should be an ASE ``Atoms`` object, with all B sites
        which are to be permuted having the same element.  If any B sites have a different element
        on them, they will not participate in the permutation.
    bs : Tuple[str, str]
        A tuple of elements to permute on the B sites, represented using their atomic symbols.
        For example:
            >>> ["Ti", "Nb"]
        The first element of the list must be the element which sits on the B sites in the passed
        model.

    Yields
    ------
    Tuple[str, Atoms]
        Each tuple consists of a label representing the B sites in the model (in the order in which
        they occur) followed by an ASE ``Atoms`` object with the B sites substituted as the label
        would suggest.  The order in which label-model pairs are yielded is arbitrary.

    """
    # Get all B atoms in model
    symbols = model.get_chemical_symbols()
    num_b_sites = symbols.count(bs[0])
    b_idxs = [idx for idx, sym in enumerate(symbols) if sym == bs[0]]

    # Generate all possible arrangements of the two B site elements for this thickness
    # These are generators because for larger models this structures could become very large
    b_combos = combinations_with_replacement(bs, num_b_sites)
    b_perms = (set(permutations(combo)) for combo in b_combos)
    b_arrs = (arrangement for subset in b_perms for arrangement in subset)

    # Yield a model for each arrangement
    for arrangement in b_arrs:
        for counter, b_idx in enumerate(b_idxs):
            symbols[b_idx] = arrangement[counter]
        new_model = deepcopy(model)
        new_model.set_chemical_symbols(symbols)
        encoding = list(zip(*reversed(rle.encode([elem for elem in arrangement]))))
        label = "".join(str(cnt)+str(sym) for cnt, sym in encoding)
        yield label, new_model


# %% SIMULATE
# Detector setup
haadf = AnnularDetector(inner=prms["haadf_min_angle"],
                        outer=prms["haadf_max_angle"])
df4 = SegmentedDetector(inner=prms["df4_min_angle"],
                        outer=prms["df4_max_angle"],
                        nbins_radial=1, nbins_angular=4,
                        rotation=prms["df4_rotation"])
detectors = [haadf, df4]

# Probe setup
probe = Probe(energy=prms["beam_energy"],
              semiangle_cutoff=prms["conv_angle"],
              device="cpu")

# Set export path here (needed in the loop), make the directory if it doesn't exist
export_path = os.path.join(prms["export_path"], "arrangements")
if not os.path.exists(export_path):
    os.makedirs(export_path)

# Build the base unit cell and make it into a surface with the correct zone axis
for za in prms["zas"]:
    print(f"Starting simulation for zone axis {''.join([str(e) for e in za])}...")
    uc = Atoms("BaTiO3",
               cell=[prms["lattice_a"], prms["lattice_a"], prms["lattice_a"]],
               pbc=True,
               scaled_positions=[(0, 0, 0),
                                 (0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0),
                                 (0.5, 0, 0.5),
                                 (0, 0.5, 0.5)])
    surf = abuild.surface(uc, za, layers=1, periodic=True)

    # Figure out what the thickness steps will be (add one projected cell each time)
    c = surf.cell[2][2]
    # Ceiling the number of steps so that the thickness set in prms will be reached
    steps = int(np.ceil(prms["thickness"] / c))
    thickness_multipliers = [(step+1) for step in range(steps)]
    for thick_mul in thickness_multipliers:
        print(f"Simulating stack {thick_mul} of {len(thickness_multipliers)} "
              f"(stack thickness {c*thick_mul} A):")
        # Generate all B-site arrangements for each thickness
        stack = surf * (1, 1, thick_mul)
        arrangements = gen_arrangements(stack, ("Ti", "Zr"))

        # Build frozen phonon configurations for each arrangement
        arrangements_with_fps = []
        for label, arr in arrangements:
            fp = FrozenPhonons(arr,
                               sigmas=prms["fp_sigmas"],
                               num_configs=prms["fp_cfgs"],
                               seed=prms["seed"])
            arrangements_with_fps.append((label, fp))

        # One frozen phonon object with n configs per arrangement
        for label, arr_fp in tqdm(arrangements_with_fps,
                                  desc="Arrangements", unit="arr"):
            potential = Potential(arr_fp,
                                  sampling=prms["sampling"],
                                  device="cpu",
                                  storage="cpu",
                                  projection="infinite",
                                  parametrization="kirkland",
                                  slice_thickness=prms["slice_thickness"])
            # Match probe gpts to potential
            probe.grid.match(potential)
            # Build the scan grid to the potential extents, slightly better than nyquist
            grid = GridScan(start=[0, 0], end=potential.extent,
                            sampling=probe.ctf.nyquist_sampling*0.9)
            measurements = probe.validate_scan_measurements(detectors, grid)

            for indices, positions in grid.generate_positions(max_batch=prms["max_batch"]):
                waves = probe.build(positions)
                # Multislice propogation
                waves = waves.multislice(potential, pbar=False)

                for detector in detectors:
                    new_measurements = detector.detect(waves)
                    # Detector measurements are per fp config, so take the mean when adding
                    grid.insert_new_measurement(measurements[detector],
                                                indices, new_measurements.mean(0))


#     # stack = []
#     # for i in range(len(measurements)):
#     #     export_name = (os.path.splitext(filename)[0] + "_PACBED_" + str(za_idx)
#     #                    + "_" + str(int(thickness)) + "A"
#     #                    + "_with_stepsize_" + str(int(thickness_step)) + "A"
#     #                    + ".tif")  # Export to tif format
#     #     xy = measurements[i][detectors[0]].sum((0, 1)).array
#     #     stack.append(xy)
#     #     tifffile.imwrite(os.path.join(export_path, export_name),
#     #                      stack, photometric='minisblack')
#     # print("Done!")

# print("\nSimulation complete!")
