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
import cupy
import os
import warnings
import tifffile
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
from copy import copy, deepcopy
from itertools import combinations_with_replacement, permutations
from random import seed, choices
from typing import Dict, List, Tuple, Set

# %% SETTINGS
prms = {"seed":            42,             # Pseudo-random seed
        # POTENTIAL SETTINGS
        "lattice_a":       4.083,          # Pseudo-cubic lattice parameter (A)
        "sampling":        0.04,           # Sampling of potential (A)
        # Total model thickness (A); will be rounded up to a full projected cell
        "thickness":       200,
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

gpu = cupy.cuda.Device(0)  # Change to device 1 before simulation if GPU0 is being used
# %% CUSTOM BUILD FUNCTIONS


def randomize_chem(atoms: Atoms,
                   replacements: Dict[str, Dict[str, float]],
                   prseed: int = prms["seed"]) -> Atoms:
    """Randomize the chemistry of an ASE ``Atoms`` object via to user-defined replacement rules.

    Parameters
    ----------
    atoms : Atoms
        Initial ASE ``Atoms`` object.  This object will be mutated.
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
    pr_seed : int, optional
        Pseudo-random seed.  The default is ``prms["seed"]``, the global pseudo-random seed for the
        script.

    Returns
    -------
    Atoms
        ASE ``Atoms`` object based on ``atoms``, but with the specified elemental replacements.
    """
    seed(prseed)

    # Sanity check:
    for elem, rep in replacements.items():
        if sum(rep.values()) < 1:  # Add in the "NOP weights" (chance to not replace) if needed
            rep[elem] = 1 - sum(rep.values())
        assert sum(rep.values()) == 1  # If this is ever False, we're likely to get garbage results

    symbols = atoms.get_chemical_symbols()
    counts = dict(zip(set(symbols), [symbols.count(e) for e in set(symbols)]))

    for elem, reps in replacements.items():
        elem_idxs = [idx for idx, sym in enumerate(symbols) if sym == elem]
        rep_with = choices(list(reps.keys()), weights=list(reps.values()), k=counts[elem])
        for i in elem_idxs:
            symbols[i] = rep_with.pop()

    atoms.set_chemical_symbols(symbols)
    return atoms


def gen_BPerm_models(model: Atoms, bs: Tuple[str, str]) -> Tuple[str, Atoms]:
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
    b_combos = combinations_with_replacement(*bs, num_b_sites)
    b_perms = (set(permutations(combo)) for combo in b_combos)
    b_arrs = (arrangement for subset in b_perms for arrangement in subset)

    # Yield a model for each arrangement
    for arrangement in b_arrs:
        for counter, b_idx in enumerate(b_idxs):
            symbols[b_idx] = arrangement[counter]
        new_model = deepcopy(model)
        new_model.set_chemical_symbols(symbols)
        label = "".join([elem for elem in arrangement])
        yield label, new_model


# %% SIMULATION LOOP

with gpu:
    # Manual GPU memory management per loop
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()

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
                  semiangle_cutoff=prms["convergence"],
                  device="gpu")

    # Build the base unit cell and make it into a surface with the correct zone axis
    for za in prms["zas"]:
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
        steps = np.ceil(prms["thickness"] / c)
        thicknesses = [(step+1)*c for step in range(steps)]
        for thickness in thicknesses:
            # Generate all B-site arrangements for each thickness
            stack = surf * (1, 1, thickness)
            arrangements = gen_BPerm_models(stack, ("Ti", "Zr"))
            # Build frozen phonon configurations for each arrangement
            fps = []
            for label, arr in arrangements:
                fp = FrozenPhonons(arr,
                                   sigmas=prms["fp_sigmas"],
                                   num_configs=prms["fp_configs"],
                                   seed=prms["seed"])
                fps.append((label, fp))

            for label, fp_config in fps:
                # Rebuild the potential for each frozon phonon configuration
                potential = Potential(fp_config,
                                      sampling=prms["sampling"],
                                      device="gpu",
                                      storage="cpu",
                                      projection="infinite",
                                      parametrization="kirkland",
                                      slice_thickness=prms["slice_thickness"])
                # Match probe gpts to potential
                probe.grid.match(potential)
                # Build the scan grid to the potential extents, slightly better than nyquist
                grid = GridScan(start=[0, 0], end=potential.extent,
                                sampling=probe.ctf.nyquist_sampling*0.9)


#     measurements = probe.validate_scan_measurements(detectors, grid)
#     # %% RUN
#     print("Beginning simulation...", end="\n\n")
#     start_time = timer()
#     num_fp_cfgs = len(fp)
#     cfg_num = 0
#     for atom_cfg in fp:
#         cfg_num += 1
#         print("Frozen phonon configuration " + str(cfg_num) + "/" + str(num_fp_cfgs) + ":")
#         # Must rebuild the potential for each frozon phonon configuration
#         potential = Potential(atom_cfg,
#                               sampling=prms["sampling"],
#                               device="gpu",
#                               storage="cpu",
#                               projection="infinite",
#                               parametrization="kirkland",
#                               slice_thickness=prms["slice_thickness"])

#     end_time = timer()
#     elapsed = "{:0.2f}".format(end_time - start_time) + "s"
#     print("Finished, elapsed time was " + elapsed)

#     # %% EXPORT
#     # TODO: Export stuff is broken now since it wanted the import filename, fix it
#     # print("Exporting...", end=" ")
#     # export_path = os.path.join(path, "PACBED")
#     # if not os.path.exists(export_path):
#     #     os.makedirs(export_path)

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
#     # mempool.free_all_blocks()
#     # pinned_mempool.free_all_blocks()

# print("\nSimulation complete!")
