from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, Sequence

from abtem import Probe, GridScan, Measurement
from abtem.structures import orthogonalize_cell
from ase import Atoms
from ase.io import cif
from abtem.detect import PixelatedDetector, FlexibleAnnularDetector, SegmentedDetector, AnnularDetector
from abtem.potentials import Potential
from abtem.temperature import FrozenPhonons
from numpy import ndarray, linalg, pi, sin, cos, degrees
from cupy import get_default_pinned_memory_pool, get_default_memory_pool
from cupy.cuda import Device, MemoryPool, PinnedMemoryPool
from ase.build import surface
from tifffile import tifffile


@dataclass(kw_only=True)
class PotentialParameters:
    """Parameters for the generation of AbTEM potentials and FrozenPhonon configurations

    Attributes
    ----------
    sampling
        Lateral sampling of the potential, 1/A
    slice_thickness
        Thickness of potential slices, A
    parametrization
        Either "lobato" or "kirkland". "lobato" is what abTEM considers to be the default, but only
        "kirkland" works with infinite projection
    projection
        Either "finite" or "infinite". "infinite" assignes the potential of each atom to a single slice,
        whereas "finite" integrates the potential between slices. "infinite" is faster and usually has
        minimal impact on accuracy
    precalculate
        If True, precalculate and store the potential.  If False, calculate it on-the-fly.  Setting this to
        False may be necessary for potentials too large to fit in memory
    device
        Either "gpu" (to compute the potential on the GPU) or "cpu" (to compute on the CPU)
    storage
        Either "gpu" (to store the potential in GPU memory) or "cpu" (to store it in RAM). GPU memory is faster,
        but RAM will usually have much more available storage space.  You can set this to None (or leave it
        unset) to default to whatever was selected for device
    seed
        Fixed seed for FrozenPhonon generation, or None to use a new seed each time the FrozenPhonons are built
    fp_cfgs
        The number of FrozenPhonon configurations to generate; this must be set if you want to generate
        FrozenPhonon configurations
    fp_sigmas
        A dictionary mapping from elements to their sqrt(U_iso), A; this must be set if you want to generate
        FrozenPhonon configurations
    """
    sampling: float | Sequence[float]  # abTEM doesn't strictly require this parameter, but I do
    slice_thickness: float
    parametrization: Literal["kirkland", "lobato"] = "kirkland"
    projection: Literal["finite", "infinite"] = "infinite"
    precalculate: bool = True
    device: Literal["gpu", "cpu"] = "gpu"
    storage: Literal["gpu", "cpu"] | None = None
    seed: int | None = None
    fp_cfgs: int | None = None
    # I'm making the typing for fp_sigmas more restrictive than abTEM's typing: for me, it must be a dict
    fp_sigmas: dict[str | int, float] | None = None

    def __post_init__(self):
        # Reimplementation of the default abTEM behavior for this wrapper
        if self.storage is None:
            self.storage = self.device


@dataclass(kw_only=True)
class ProbeParameters:
    """Parameters for the generation of abTEM Probes, implementing defaults for a typical perfect probe

    Attributes
    ----------
    max_batch
        The maximum number of probe positions to propogate at once (default=50)
    energy
        The energy of the incident beam, keV; equivalently, the accelerating volatage, kV (default=200E3)
    convergence
        The semiangle of convergence of the probe, mrad (default=17.9)
    device
        Either "gpu" (to run the simulation on the GPU) or "cpu" (to run on the CPU)
    tilt_mag
        The probe tilt, mrad (default=0)
    tilt_rot
        The probe tile rotation from +x axis, rad (default=0)
    tilt
        The probe tilt represented in x and y components
    defocus
        Probe defocus C1, A (default=0)
    stig
        Probe 2-fold stigmation A1, A (default=0)
    stig_rot
        Rotation of A1, rad (default=0)
    coma
        Probe 2nd order axial coma B2, A (default=0)
    coma_rot
        Rotation of B2, rad (default=0)
    spherical
        Probe spherical abbreation C3, A (default=0)
    """
    max_batch: int = 50
    energy: float = 200E3
    convergence: float = 17.9
    device: Literal["gpu", "cpu"] = "gpu"
    tilt_mag: float = 0
    tilt_rot: float = 0
    defocus: float = 0
    stig: float = 0
    stig_rot: float = 0
    coma: float = 0
    coma_rot: float = 0
    spherical: float = 0

    def __post_init__(self):
        self.tilt = (self.tilt_mag * sin(self.tilt_rot),
                     self.tilt_mag * cos(self.tilt_rot))


@dataclass(kw_only=True)
class DetectorBuilder:
    """This class is basically just a way to hold default detector parameters that correspond to the Themis.
    It has a few class methods to return detectors.

    Attributes
    ----------
    save_path
        The path to automatically save Measurements made by the detectors.  Filenames will be generated
        automatically based on the detector type.
    annular_inner_angle
        Annular detector inner angle, mrad.  Default is 69
    annular_outer_angle
        Annular detector outer angle, mrad.  Default is 200
    segmented_inner_angle
        Segmented detector inner angle, mrad.  Default is 16
    segmented_outer_angle
        Segmented detector outer angle, mrad.  Default is 65
    segmented_rot
        Segmented detector rotation, rad.  Default is 1.8*pi = 324 deg
    pixelated_max_angle
        Pixelated detector maximum angle, mrad.  Default is 40
    """
    save_path: str | PathLike | None = None
    annular_inner_angle: float = 69
    annular_outer_angle: float = 200
    segmented_inner_angle: float = 16
    segmented_outer_angle: float = 65
    segmented_rot: float = 1.8 * pi
    pixelated_max_angle: float = 40

    def _save_file_name(self, name: str):
        if self.save_path is not None:
            save_file = str(Path(self.save_path) / name)
        else:
            save_file = None
        return save_file

    def get_haadf(self):
        return AnnularDetector(inner=self.annular_inner_angle,
                               outer=self.annular_outer_angle,
                               save_file=self._save_file_name("haadf"))

    def get_df4(self):
        return SegmentedDetector(inner=self.segmented_inner_angle,
                                 outer=self.segmented_outer_angle,
                                 nbins_radial=1,
                                 nbins_angular=4,
                                 save_file=self._save_file_name("df4"))

    def get_pixelated(self):
        return PixelatedDetector(max_angle=self.pixelated_max_angle,
                                 resample="uniform",
                                 save_file=self._save_file_name("pixelated"))

    def get_flexible(self):
        return FlexibleAnnularDetector(save_file=self._save_file_name("flex"))


@dataclass
class Model:
    """A wrapper around an ASE atoms object that holds a bit of useful information.

    Attributes
    ----------
    atoms
        The ASE atoms object at the core of the model.
    a, b, c
        A vector spanning one of the dimensions of the projected unit cell; unlike ASE's internal representation,
        this does not change if you repeat the cell along an axis
    za
        The zone axis of the projected cell making up the model; set to None if that concept isn't meaningful for
        your particular model
    thickness
        The total thickness of the model along the beam direction (you set this manually since this could have
        many valid values for a single model, depending on how you rotate it before feeding it to abTEM)
    name
        A string to name the model; may be used in file name generation, otherwise just useful for reference
    """
    atoms: Atoms
    a: float
    b: float
    c: float
    za: tuple[int, int, int] | None
    thickness: float
    name: str | None = None


@dataclass
class PACBEDResult:
    """Holds the results of a PACBED simulation in a convenient format

    Attributes
    ----------
    model
        The model used in the simulation
    potential_params
        The PotentialParameters class used to generate Potentials / FrozenPhonons from the model
    probe_params
        The ProbeParameters class used in the simulation
    measurements
        The results of the simulation
    thicknesses
        The thickness at index i corresponds to the measurement at the same index in its array
    detector
        The PixelatedDetector used to capture the PACBEDs; this is needed to index into the measurements
    """
    model: Model
    potential_params: PotentialParameters
    probe_params: ProbeParameters
    measurements: list[ndarray]
    thicknesses: list[float]
    detector: PixelatedDetector


@dataclass
class STEMResult:
    """Holds the results of a STEM simulation in a convenient format

    Attributes
    ----------
    model
        The model used in the simulation
    potential_params
        The PotentialParameters class used to generate Potentials / FrozenPhonons from the model
    probe_params
        The ProbeParameters class used in the simulation
    grid
        GridScan object associated with the STEM scan; needed to interpolate results properly
    measurement
        The results of the simulation
    """
    model: Model
    potential_params: PotentialParameters
    probe_params: ProbeParameters
    grid: GridScan
    measurement: Measurement | list[Measurement]
    images: list[ndarray] | None = None

    def __post_init__(self):
        # TODO: This isn't working for some reason
        images = []
        if type(self.measurement) is not list:
            self.measurement = [self.measurement]
        for meas in self.measurement:
            if len(meas.array.shape) == 2:
                images.append(meas.array)
        if len(images) != 0:
            self.images = images


def gen_potentials(model: Atoms,
                   prms: PotentialParameters)\
        -> Potential:
    """Wrapper for generating abTEM Potentials"""
    return Potential(model,
                     sampling=prms.sampling,
                     device=prms.device,
                     storage=prms.storage,
                     projection=prms.projection,
                     parametrization=prms.parametrization,
                     precalculate=prms.precalculate,
                     slice_thickness=prms.slice_thickness)


def gen_phonons(model: Atoms,
                prms: PotentialParameters)\
        -> FrozenPhonons:
    """Wrapper for generating abTEM FrozenPhonons"""
    if prms.fp_cfgs is None or prms.fp_sigmas is None:
        raise ValueError("Both `fp_cfgs` and `fp_sigmas` must be set in order to generate FrozenPhonons")
    return FrozenPhonons(model,
                         sigmas=prms.fp_sigmas,
                         num_configs=prms.fp_cfgs,
                         seed=prms.seed)


def get_span(model: Atoms,
             za: tuple[int, int, int])\
        -> tuple[float, float, float]:
    """Get the spanning vector magnitudes for a unit cell along a certain projection.

    Parameters
    ----------
    model
        Atoms object to compute the projected cell for; usually it makes the most sense if this is a unit cell
    za
        The zone axis along which to project the model (relative to its default basis vectors)
    Returns
    -------
    a, b, c
        A tuple of the magnitudes of the vectors that span the projected cell, in the order: a, b, c (or,
        equivalently, x, y, z)
    """
    _tmp_atoms = surface(model, indices=za, layers=1, periodic=True)
    _a, _b, _c = _tmp_atoms.cell
    a, b, c = linalg.norm(_a), linalg.norm(_b), linalg.norm(_c)
    return a, b, c


def gpu_setup(gpu_num: int = 0)\
        -> tuple[Device, MemoryPool, PinnedMemoryPool]:
    """If the simulation is going to be run on the GPU, run this."""
    mempool = get_default_memory_pool()
    pinned_mempool = get_default_pinned_memory_pool()
    dev = Device(gpu_num)
    return dev, mempool, pinned_mempool


def free_gpu_memory(mempool: MemoryPool,
                    pinned_mempool: PinnedMemoryPool)\
        -> None:
    """Run this to free GPU memory without restarting the console if simulating on the GPU"""
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


def build_from_cif(cif_path: str | PathLike,
                   zas: tuple[int, int, int] | list[tuple[int, int, int]],
                   tks: float | list[float],
                   tilings: tuple[int, int] | list[tuple[int, int]])\
        -> list[Model]:
    """Generate one or more ASE atoms objects, and package them as models

    Parameters
    ----------
    cif_path
        The path to a cif file, or a string representation of that path
    zas
        Zone axes should be tuples (e.g. (1, 0, 1)).  A single zone axis can be passed, or multiple can be
        packaged in a list (which will increase the number of models generated)
    tks
        The thickness of the model (i.e. along the beam direction), in A.  If multiple thicknesses are passed,
        multiple models will be generated
    tilings
        A tuple of x and y tiling, in multiples of the projected unit cell; if zas is a list, this should be too,
        and of the same length.  The order determines the tiling of the corresponding za

    Returns
    -------
    models
        A list of models (ASE Atoms objects); this will always be packed into a list, even if only one model
        was generated
    """
    # Reading from a cif file returns a generator, because cif files can hold multiple structures
    # So we'll make a list of them and iterate over them all!  (Usually there will only be one.)
    with open(Path(cif_path)) as ciffile:
        cif_cells = list(cif.read_cif(ciffile, slice(None)))

    models: Model | list[Model] = []
    if ((type(zas) is list) != (type(tilings) is list))\
            or ((type(zas) is list and type(tilings) is list) and len(zas) != len(tilings)):
        raise ValueError("Must pass exactly one tiling per zone axis")
    if type(zas) is not list and type:  # If we only got one zone axis and one tiling, wrap them each in a list
        zas = [zas]
        tilings = [tilings]

    for cif_cell in cif_cells:
        for za, til in zip(zas, tilings):
            if type(tks) is not list:
                tks = [tks]
            for thickness in tks:
                a, b, c = get_span(cif_cell, za)
                thickness_multiplier = int((thickness // c)) + 1
                atoms: Atoms = surface(cif_cell, indices=za, layers=thickness_multiplier, periodic=True)
                atoms = orthogonalize_cell(atoms)
                atoms *= (til[0], til[1], 1)
                new_model = Model(atoms, a, b, c, za, thickness)
                models.append(new_model)
    return models


def simulate_packbed_thickness_series(models: list[Model],
                                      det: DetectorBuilder,
                                      potential_prms: PotentialParameters,
                                      probe_prms: ProbeParameters,
                                      thickness_step: float)\
        -> list[PACBEDResult]:
    if potential_prms.device == "gpu" or probe_prms.device == "gpu":
        dev, mempool, pinned_mempool = gpu_setup(0)
    else:
        dev = Device()  # Default is None, which I believe sets the device to the CPU

    with dev:
        detector = det.get_pixelated()
        results: list[PACBEDResult] = []
        for model in models:
            # Initial atom potential for grid matching; won't be directly used in sims
            potential = Potential(model.atoms,
                                  sampling=potential_prms.sampling,
                                  device=potential_prms.device,
                                  projection=potential_prms.projection,
                                  parametrization=potential_prms.parametrization,
                                  slice_thickness=potential_prms.slice_thickness)

            probe = Probe(energy=probe_prms.energy,
                          semiangle_cutoff=probe_prms.convergence,
                          device=probe_prms.device,
                          tilt=probe_prms.tilt)
            probe.ctf.set_parameters({"astigmatism": probe_prms.stig,
                                      "astigmatism_angle": probe_prms.stig_rot,
                                      "coma": probe_prms.coma,
                                      "coma_angle": probe_prms.coma_rot,
                                      "Cs": probe_prms.spherical})
            probe.grid.match(potential)

            grid = GridScan(start=[0, 0],
                            end=[model.a, model.b],
                            sampling=probe.ctf.nyquist_sampling)

            fp = FrozenPhonons(model.atoms,
                               sigmas=potential_prms.fp_sigmas,
                               num_configs=potential_prms.fp_cfgs,
                               seed=potential_prms.seed)

            save_delta = round(thickness_step / potential_prms.slice_thickness)
            save_chunks = [(i, i + save_delta) for i in range(0, len(potential), save_delta)]
            measurements = [probe.validate_scan_measurements([detector], grid) for _ in save_chunks]

            del potential  # Don't accidentally use this later, we want to use FrozenPhonons now that they exist

            for atom_cfg in fp:
                # Must rebuild the potential for each frozen phonon configuration
                potential = Potential(atom_cfg,
                                      sampling=potential_prms.sampling,
                                      device=potential_prms.device,
                                      storage=potential_prms.storage,
                                      projection=potential_prms.projection,
                                      parametrization=potential_prms.parametrization,
                                      slice_thickness=potential_prms.slice_thickness)

                for indices, positions in grid.generate_positions(max_batch=probe_prms.max_batch):
                    waves = probe.build(positions)
                    for chunk_idx, (slice_start, slice_end) in enumerate(save_chunks):
                        potential_slices = potential[slice_start:slice_end]
                        waves = waves.multislice(potential_slices, pbar=False)
                        new_measurements = detector.detect(waves)
                        grid.insert_new_measurement(measurements[chunk_idx][detector],
                                                    indices, new_measurements)

            thicknesses = []
            for (_, slice_end) in save_chunks:
                thicknesses.append(round(slice_end / potential_prms.slice_thickness * model.c / 10, 2))
            result = PACBEDResult(model=model,
                                  potential_params=potential_prms,
                                  probe_params=probe_prms,
                                  measurements=measurements,
                                  thicknesses=thicknesses,
                                  detector=detector)
            results.append(result)

    # Free the GPU memory for others; if this isn't done, GPU memory is only freed when the Python kernel is reset
    if potential_prms.device == "gpu" or probe_prms.device == "gpu":
        # noinspection PyUnboundLocalVariable
        free_gpu_memory(mempool, pinned_mempool)
    return results


def export_pacbed_as_tif(results: list[PACBEDResult],
                         export_path: str | PathLike,
                         probe_prms: ProbeParameters,
                         thickness_step: float)\
        -> None:
    export_path = Path(export_path)
    try:
        export_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as ferror:
        raise ferror
    for res in results:
        stack = []
        if res.model.name:
            name = res.model.name
        else:
            name = ""
        export_name = (f"{name}_PACBED_" +
                       f"tilt{probe_prms.tilt_mag}mrad@{degrees(probe_prms.tilt_rot)}deg_" +
                       f"{str(res.model.za)}_{str(int(res.model.thickness))}A_with_stepsize" +
                       f"{str(int(thickness_step))}A.tif")

        for i in range(len(res.measurements)):
            xy = res.measurements[i][res.detector].sum((0, 1)).array
            stack.append(xy)

        tifffile.imwrite(export_path / export_name, stack, photometric="minisblack")


def sim_stem(models: list[Model],
             det: DetectorBuilder,
             potential_prms: PotentialParameters,
             probe_prms: ProbeParameters,
             kinds: list[Literal["flex", "flexible", "haadf", "df4", "dpc", "seg", "segmented"]] = None)\
        -> list[STEMResult]:
    """Run a standard STEM simulation with a given model (or list of models)

    Parameters
    ----------
    models
        The model or models to run the simulation on
    det
        Builds a detector with given parameters, based on the
    potential_prms
    probe_prms
    kinds

    Returns
    -------

    """
    if kinds is None:
        kinds = ["flex"]
    if potential_prms.device == "gpu" or probe_prms.device == "gpu":
        dev, mempool, pinned_mempool = gpu_setup(0)
    else:
        dev = Device()  # Default is None, which I believe sets the device to the CPU

    with dev:
        detectors = []
        for k in kinds:
            if k in ["flex", "flexible"]:
                detectors.append(det.get_flexible())
            elif k in ["haadf"]:
                detectors.append(det.get_haadf())
            elif k in ["df4", "dpc", "seg", "segmented"]:
                detectors.append(det.get_df4())
            else:
                raise ValueError("Unknown simulation kind; see the documentation")

        results: list[STEMResult] = []
        for model in models:
            probe = Probe(energy=probe_prms.energy,
                          semiangle_cutoff=probe_prms.convergence,
                          device=probe_prms.device,
                          tilt=probe_prms.tilt)

            probe.ctf.set_parameters({"astigmatism": probe_prms.stig,
                                      "astigmatism_angle": probe_prms.stig_rot,
                                      "coma": probe_prms.coma,
                                      "coma_angle": probe_prms.coma_rot,
                                      "Cs": probe_prms.spherical})

            fp = FrozenPhonons(model.atoms,
                               sigmas=potential_prms.fp_sigmas,
                               num_configs=potential_prms.fp_cfgs,
                               seed=potential_prms.seed)

            potential = Potential(fp,
                                  sampling=potential_prms.sampling,
                                  device=potential_prms.device,
                                  projection=potential_prms.projection,
                                  parametrization=potential_prms.parametrization,
                                  slice_thickness=potential_prms.slice_thickness)

            probe.grid.match(potential)

            grid = GridScan(start=[0, 0],
                            end=potential.extent,
                            sampling=probe.ctf.nyquist_sampling)

            measurements = probe.scan(grid, detectors, potential, pbar=False)

            results.append(STEMResult(model,
                                      potential_prms,
                                      probe_prms,
                                      grid,
                                      measurements))
    # Free the GPU memory for others; if this isn't done, GPU memory is only freed when the Python kernel is reset
    if potential_prms.device == "gpu" or probe_prms.device == "gpu":
        # noinspection PyUnboundLocalVariable
        free_gpu_memory(mempool, pinned_mempool)
    return results
