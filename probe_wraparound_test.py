# %% IMPORTS

# ASE Imports
from ase import Atoms
import ase.build as abuild
# AbTEM Imports
from abtem.potentials import Potential
from abtem.structures import orthogonalize_cell
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.detect import WavefunctionDetector
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
    """Simple helper function to generate potentials in a list comprehension

    Parameters
    ----------
    model : FrozenPhonons
        abTEM ``FrozenPhonons`` object.

    Returns
    -------
    Potential
        abTEM ``Potential`` object.

    """

    return Potential(model,
                     sampling=0.04,
                     device="gpu",
                     storage="cpu",
                     projection="infinite",
                     parametrization="kirkland",
                     slice_thickness=1)


# %% SETUP

# FA detector will let us easily test different camera lengths without redoing the sim
detector = WavefunctionDetector()

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

# 10 to 20nm in 1nm steps (as near as possible)
n_ucs = [25, 27, 29, 32, 34, 37, 39, 42, 44, 47, 49]
stacks = [atoms * (2, 2, n) for n in n_ucs]  # 2x2 needed to avoid probe wraparound

# Prebuild potentials
potentials = [_gen_potentials(stack) for stack in stacks]

# Single scan point for grid to check for probe wraparound
scanpoint = GridScan(start=[0, 0], end=[0.1, 0.1], sampling=0.1)

# %% SIMULATE

images = []  # Will hold the image data from each probe for plotting

conv_angles = [14, 16, 18, 20]
for ca in conv_angles:
    print(f"Simulating with probe convergence angle {ca}mrad...")
    with cupy.cuda.Device(1):
        probe = Probe(energy=200E3,
                      semiangle_cutoff=ca,
                      device="gpu")
        probe.ctf.set_parameters({"astigmatism": 7,
                                  "astigmatism_angle": 155,
                                  "coma": 300,
                                  "coma_angle": 155,
                                  "Cs": -5000})
        for potential in tqdm(potentials):
            probe.grid.match(potential)
            measurement = probe.scan(scanpoint, [detector], potential, pbar=False)

            exit_wave = [np.abs(x)**2 for x in measurement.array[0, 0, :, :]]
            images.append(exit_wave)

# %% VIZ

_, axes = plt.subplots(4, 11, figsize=(10, 10))
axes = axes.flatten()

labels = []
for ca in conv_angles:
    for thk in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        labels.append(f"{ca}mrad @ {thk}nm")

for img, lab, ax in zip(images, labels, axes):
    ax.imshow(img, cmap="cividis")
    ax.set_title(lab)
    ax.axis("off")
plt.show()
