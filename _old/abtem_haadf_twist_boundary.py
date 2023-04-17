# %% IMPORTS

# ASE Imports
import ase.io as aio
import ase.build as abuild
from ase.visualize import view as aviz
# AbTEM Imports
from abtem.potentials import Potential
from abtem.waves import SMatrix
from abtem.scan import GridScan
from abtem.detect import AnnularDetector
from abtem.detect import SegmentedDetector
from abtem.temperature import FrozenPhonons
# Other Imports
from cupy.fft.config import get_plan_cache, show_plan_cache_info
from cupy import cuda
import os
from timeit import default_timer as timer
import warnings
from tqdm import tqdm
import tifffile
import numpy as np

# %% SETTINGS
prms = {"seed":            42,              # Pseudo-random seed
        # STRUCTURE FILE LOCATION
        "path":            r"E:\Users\Charles\Twist_Boundary",
        "filename":        "twist25l8.cif",
        # POTENTIAL SETTINGS
        "sampling":        0.02,            # Sampling of potential (A)
        "slice_thickness": 1,               # Thickness per slice (A)
        "fp_cfgs":         2,               # Number of frozen phonon configurations to simulate
        "fp_sigmas":       {"Ti": 0.0789,   # Frozen phonon sigma values per atom type
                            "O":  0.0720},  # Data from http://ruby.colorado.edu/~smyth/Research/Papers/Swope_p448-453_95.pdf
        # PROBE SETTINGS
        "beam_energy":     200E3,           # Energy of the electron beam (eV)
        "conv_angle":      17.9,            # Probe semiangle of convergence (mrad)
        "interpolation":   4,               # Probe interpolation factor for PRISM
        # DETECTOR SETTINGS
        "haadf_min_angle": 69,              # HAADF annulus inner extent (mrad)
        "haadf_max_angle": 200,             # HAADF annulus outer extent (mrad)
        "scan_x":          2,               # Scan partitions along the x-axis of the potential
        "scan_y":          2}               # Scan partitions along the y-axis of the potential

gpu = cuda.Device(0)  # Change to device 1 before simulation if GPU0 is being used
cache = get_plan_cache()
cache.set_size(0)
# %% IMPORT STRUCTURE

struct = aio.read(os.path.join(prms["path"], prms["filename"]))
# aviz(struct)

proj_cell = abuild.surface(struct, indices=(0, 0, 1), layers=1, periodic=True)
# aviz(proj_cell)

# %% SETUP
print("Setting up simulation...", end=" ")
start_time = timer()

potential = Potential(proj_cell,
                      sampling=prms["sampling"],
                      device="gpu",
                      storage="cpu",
                      projection="infinite",
                      parametrization="kirkland",
                      precalculate=False)

probe = SMatrix(energy=prms["beam_energy"],
                semiangle_cutoff=prms["conv_angle"],
                interpolation=prms["interpolation"],
                device="gpu",
                storage="cpu")

haadf = AnnularDetector(inner=prms["haadf_min_angle"],
                        outer=prms["haadf_max_angle"])

grid = GridScan(start=[0, 0],  # Grid whole sim cell
                end=potential.extent,
                sampling=probe.ctf.nyquist_sampling)

probe = probe.multislice(potential, pbar=False)
probe = probe.downsample("limit")

scans = grid.partition_scan((prms["scan_x"], prms["scan_y"]))
measurements = haadf.allocate_measurement(probe, grid)

end_time = timer()
elapsed = "{:0.2f}".format(end_time - start_time) + "s"
print(f"Done! Elasped time was {elapsed}")
# %% RUN
for scan in tqdm(scans, total=len(scans)):
    cropped = probe.crop_to_scan(scan)
    cropped = cropped.transfer("gpu")
    cropped.scan(scan, haadf, measurements=measurements, pbar=False)

# %% EXPORT
measurements.interpolate(0.05).show(figsize=(12, 12))
