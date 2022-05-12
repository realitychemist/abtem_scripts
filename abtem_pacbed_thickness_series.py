# %% IMPORTS

# ASE Imports
import ase.io as aio
import ase.build as abuild
# AbTEM Imports
from abtem.potentials import Potential
from abtem.waves import Probe
from abtem.scan import GridScan
from abtem.structures import orthogonalize_cell
from abtem.detect import PixelatedDetector
from abtem.temperature import FrozenPhonons
# Other Imports
import cupy
import os
from timeit import default_timer as timer
import warnings
from tqdm import tqdm
import tifffile

# %% SETTINGS
prms = {"seed":            42,              # Pseudo-random seed
        # STRUCTURE FILE LOCATION
        "path":            r"E:\Users\Charles\BTO PACBED\abtem",
        "filename":        "BaTiO3_mp-2998_conventional_standard.cif",
        # POTENTIAL SETTINGS
        "sampling":        0.2,             # Sampling of potential, A
        "tiling":          30,              # Number of times to tile projected cell
        "thickness":       200,             # Total model thickness (A)
        "thickness_step":  10,              # PACBED export thickness steps (A)
        "slice_thickness": 2,               # Thickness per simulation slice (A)
        "zas":             [(0, 0, 1),      # Zone axes to model
                            (0, 1, 1)],
        "fp_configs":      10,              # Number of frozen phonon configurations
        "fp_sigmas":       {"Ba": 0.0757,   # Frozen phonon sigma values per atom type
                            "Ti": 0.0893,   # sigma == sqrt(U_iso) (confirmed with abTEM author)
                            "Zr": 0.1050,   # Data from https://materials.springer.com/isp/crystallographic/docs/sd_1410590
                            "O":  0.0810},  # and from https://pubs.acs.org/doi/10.1021/acs.chemmater.9b04437 (for Zr)
        # PROBE SETTINGS
        "beam_energy":     200E3,           # Energy of the electron beam (eV)
        "convergence":     17.9,            # Probe semiangle of convergence (mrad)
        # DETECTOR SETTINGS
        "max_batch":       200,             # Number of probe positions to propogate at once
        "max_angle":       40}              # Maximum detector angle (mrad)

gpu = cupy.cuda.Device(0)  # Change to device 1 before simulation if GPU0 is being used
# %% RUN
struct = aio.read(os.path.join(prms["path"], prms["filename"]))
with gpu:
    # For GPU memory management
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()

    detectors = [PixelatedDetector(max_angle=prms["max_angle"], resample="uniform")]
    for za_idx in prms["zas"]:
        start_time = timer()
        print("\nSetting up for " + str(za_idx) + " zone axis...", end=" ")
        atoms = abuild.surface(struct, indices=za_idx, layers=1, periodic=True)
        atoms = orthogonalize_cell(atoms)
        a, b, c = atoms.cell
        a = a[0]
        b = b[1]
        c = c[2]
        thickness_multiplier = int((prms["thickness"] // c)) + 1
        atoms *= (prms["tiling"], prms["tiling"], prms["thickness_multiplier"])

        # Initial atom potential for grid matching; won't be directly used in sims
        potential = Potential(atoms,
                              sampling=prms["sampling"],
                              device="gpu",
                              projection="infinite",
                              parametrization="kirkland",
                              slice_thickness=prms["slice_thickness"])

        probe = Probe(energy=prms["beam_energy"],
                      semiangle_cutoff=prms["convergence"],
                      device="gpu")
        probe.grid.match(potential)

        if(any(angle < prms["max_angle"] for angle in probe.cutoff_scattering_angles)):
            warnings.warn("Scattering angle cutoffs < 200mrad; increase potential sampling")

        grid = GridScan(start=[0, 0],
                        end=[a, b],
                        sampling=probe.ctf.nyquist_sampling)

        save_delta = round(prms["thickness_step"] / prms["slice_thickness"])
        save_chunks = [(i, i + save_delta) for i in range(0, len(potential), save_delta)]
        measurements = [probe.validate_scan_measurements(detectors, grid) for chunk in save_chunks]

        fp = FrozenPhonons(atoms,
                           sigmas=prms["fp_sigmas"],
                           num_configs=prms["fp_configs"],
                           seed=prms["seed"])

        end_time = timer()
        elapsed = "{:0.2f}".format(end_time - start_time) + "s"
        print("Done!  Elapsed time was " + elapsed, end="\n\n")

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

            for indices, positions in grid.generate_positions(max_batch=prms["max_batch"]):
                waves = probe.build(positions)
                for chunk_idx, (slice_start, slice_end) in tqdm(enumerate(save_chunks),
                                                                total=len(save_chunks),
                                                                desc="Propogating",
                                                                unit="chunk"):
                    potential_slices = potential[slice_start:slice_end]
                    waves = waves.multislice(potential_slices, pbar=False)
                    for detector in detectors:
                        new_measurements = detector.detect(waves)
                        grid.insert_new_measurement(measurements[chunk_idx][detector],
                                                    indices, new_measurements)
        end_time = timer()
        elapsed = "{:0.2f}".format(end_time - start_time) + "s"
        print("Finished simulating zone axis " + str(za_idx), ", elapsed time was " + elapsed)

        # %% EXPORT
        print("Exporting...", end=" ")
        export_path = os.path.join(prms["path"], "PACBED")
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        stack = []
        for i in range(len(measurements)):
            export_name = (os.path.splitext(prms["filename"])[0] + "_PACBED_" + str(za_idx)
                           + "_" + str(int(prms["thickness"])) + "A"
                           + "_with_stepsize_" + str(int(prms["thickness_step"])) + "A"
                           + ".tif")  # Export to tif format
            xy = measurements[i][detectors[0]].sum((0, 1)).array
            stack.append(xy)
            tifffile.imwrite(os.path.join(export_path, export_name),
                             stack, photometric='minisblack')
        print("Done!")
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    print("\nSimulation complete!")
