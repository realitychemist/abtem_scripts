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
import tifffile
import cupy
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm
from copy import deepcopy
from random import seed, choices
from typing import Dict

# %% CUSTOM BUILD FUNCTIONS


def randomize_chem(atoms: Atoms,
                   replacements: Dict[str, Dict[str, float]],
                   prseed: int = 42) -> Atoms:
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
        Pseudo-random seed.  The default is ``prms["seed"]``, the global pseudo-random seed for the
        script.

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


def _gen_potentials(model):
    return Potential(model,
                     sampling=0.04,
                     device="gpu",
                     storage="cpu",
                     projection="infinite",
                     parametrization="kirkland",
                     slice_thickness=1)


def _gen_phonons(atoms, seed):
    return FrozenPhonons(atoms,
                         sigmas={"Ba": 0.0757,
                                 "Ti": 0.0893,
                                 "Zr": 0.1050,
                                 "O":  0.0810},
                         num_configs=10,
                         seed=seed)


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

stack = atoms * (2, 2, 36)  # 2x2 needed to avoid probe wraparound, 36 --> 14.7nm

# Random seeds from random.org
seeds = [87353, 39801, 56916, 62903, 76446, 40231, 92312, 43299, 72148, 37976,
         93458, 22838, 78787, 89538, 53240, 82349, 85799, 94281, 53053, 10655,
         94124, 6828, 21401, 75500, 7576, 74045, 70885, 23437, 25341, 59347]
models = [randomize_chem(stack, {"Ti": {"Zr": 0.3}}, prseed=s) for s in seeds]

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


# %% SIMULATE

with cupy.cuda.Device(1):
    for potential in tqdm(potentials, desc="Simulating", unit="cfg"):
        probe.grid.match(potential)  # Could technically happen outside loop, but it's fast & works
        measurement = probe.scan(scan, [detector], potential, pbar=False)

        exit_wave = [np.abs(x)**2 for x in measurement.array[0, 0, :, :]]
        images.append(exit_wave)
