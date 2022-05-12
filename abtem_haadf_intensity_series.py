# %% IMPORTS

# ASE Imports
from ase import Atoms
import ase.build as abuild
# AbTEM Imports
from abtem.potentials import Potential
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.detect import AnnularDetector
from abtem.detect import SegmentedDetector
from abtem.temperature import FrozenPhonons
# Other Imports
import cupy
import os
from timeit import default_timer as timer
import warnings
from typing import Dict, List, Tuple
from tqdm import tqdm
import tifffile
from copy import deepcopy
import numpy as np

# %% SETTINGS
prms = {"seed":            42,             # Pseudo-random seed
        # POTENTIAL SETTINGS
        "sampling":        0.04,           # Sampling of potential (A)
        "thickness":       200,            # Total model thickness (A)
        "slice_thickness": 1,              # Thickness per slice (A)
        "zas":            [(0, 0, 1),      # Zone axes to model and simulate
                           (0, 1, 1)],
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
# %% ASE BUILD CELL


def randomize_chem(atoms: Atoms,
                   replacements: Dict[str, Dict[str, float]],
                   pr_seed: int = prms["seed"]) -> Atoms:
    """
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
    from random import seed, choices
    seed(seed)

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


def gen_models(abx: Tuple[str, str, str],
               a: int = 4.083,
               thickness: float = prms["thickness"],
               zas: List[Tuple[int, int, int]] = prms["zas"]) -> List[Atoms]:
    """
    Parameters
    ----------
    abx : Tuple[str, str, str]
        DESCRIPTION.
    a : int, optional
        DESCRIPTION. The default is 4.083.
    thickness : float, optional
        DESCRIPTION. The default is prms["thickness"].
    zas : List[Tuple[int, int, int]], optional
        DESCRIPTION. The default is prms["zas"].

    Returns
    -------
    List[Atoms]
        DESCRIPTION.
    """

    models = None
    return models


# a = 4.083
# base_uc = Atoms("BaTiO3",
#                 cell=[a, a, a],
#                 pbc=[1, 1, 0],
#                 scaled_positions=[(0, 0, 0),
#                                   (0.5, 0.5, 0.5),
#                                   (0.5, 0.5, 0),
#                                   (0.5, 0, 0.5),
#                                   (0, 0.5, 0.5)])


# stack_001 = abuild.surface(base_uc, indices=(0, 0, 1), layers=1, periodic=True)
# thickness_multiplier = int((prms["thickness"] // a)) + 1  # Valid only for cubic 001 proj.
# stack_001 *= (1, 1, thickness_multiplier)

# all_titanium = deepcopy(stack_001)
# all_zirconium = randomize_chem(deepcopy(stack_001), {"Ti": {"Zr": 1}})
# bzt40_random = randomize_chem(deepcopy(stack_001), {"Ti": {"Zr": 0.4}})

# models = [all_titanium, all_zirconium, bzt40_random]

# %% SETUP
with gpu:
    # For GPU memory management
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()

    # Frozen phonon model configuration
    fps = []
    for model in models:
        fp = FrozenPhonons(model,
                           sigmas=prms["fp_sigmas"],
                           num_configs=prms["fp_configs"],
                           seed=prms["seed"])
        fps.append(fp)

    # Detector setup
    haadf = AnnularDetector(inner=prms["haadf_min_angle"],
                            outer=prms["haadf_max_angle"])
    df4 = SegmentedDetector(inner=prms["df4_min_angle"],
                            outer=prms["df4_max_angle"],
                            nbins_radial=1,
                            nbins_angular=4,
                            rotation=prms["df4_rotation"])
    detectors = [haadf, df4]

    # All of the potentials should be the same, so we only need to setup one probe
    # Initial atom potential for grid matching; won't be directly used in sims
    potential = Potential(models[0],  # Use whatever is in the first spot (no particular reason)
                          sampling=prms["sampling"],
                          device="gpu",
                          projection="infinite",
                          parametrization="kirkland",
                          slice_thickness=prms["slice_thickness"])

    probe = Probe(energy=prms["beam_energy"],
                  semiangle_cutoff=prms["convergence"],
                  device="gpu")
    probe.grid.match(potential)

    # Make sure that we have sufficient sampling; this should cause a simulation error if not
    #  addressed, but we want to make sure (and it's better to catch it early)
    test_angle = min(probe.cutoff_scattering_angles)
    req_angle = max(prms["haadf_max_angle"], prms["df4_max_angle"])
    if test_angle < req_angle:
        warnings.warn("Scattering angle cutoffs too small ({0} mrad < {1} mrad); "
                      "increase potential sampling".format(round(test_angle, ndigits=1),
                                                           round(req_angle, ndigits=1)))
    # Tidy namespace
    del test_angle, req_angle

    grid = GridScan(start=[0, 0],  # Grid whole sim cell
                    end=[models[0].cell[0][0], models[0].cell[1][1]],
                    sampling=probe.ctf.nyquist_sampling)
    measurements = probe.validate_scan_measurements(detectors, grid)
    # %% RUN
    print("Beginning simulation...", end="\n\n")
    start_time = timer()
    num_fp_cfgs = len(fp)
    cfg_num = 0
    for atom_cfg in fp:
        cfg_num += 1
        print("Frozen phonon configuration " + str(cfg_num) + "/" + str(num_fp_cfgs) + ":")
        # Must rebuild the potential for each frozon phonon configuration
        potential = Potential(atom_cfg,
                              sampling=prms["sampling"],
                              device="gpu",
                              storage="cpu",
                              projection="infinite",
                              parametrization="kirkland",
                              slice_thickness=prms["slice_thickness"])

    end_time = timer()
    elapsed = "{:0.2f}".format(end_time - start_time) + "s"
    print("Finished, elapsed time was " + elapsed)

    # %% EXPORT
    # TODO: Export stuff is broken now since it wanted the import filename, fix it
    # print("Exporting...", end=" ")
    # export_path = os.path.join(path, "PACBED")
    # if not os.path.exists(export_path):
    #     os.makedirs(export_path)

    # stack = []
    # for i in range(len(measurements)):
    #     export_name = (os.path.splitext(filename)[0] + "_PACBED_" + str(za_idx)
    #                    + "_" + str(int(thickness)) + "A"
    #                    + "_with_stepsize_" + str(int(thickness_step)) + "A"
    #                    + ".tif")  # Export to tif format
    #     xy = measurements[i][detectors[0]].sum((0, 1)).array
    #     stack.append(xy)
    #     tifffile.imwrite(os.path.join(export_path, export_name),
    #                      stack, photometric='minisblack')
    # print("Done!")
    # mempool.free_all_blocks()
    # pinned_mempool.free_all_blocks()

print("\nSimulation complete!")
