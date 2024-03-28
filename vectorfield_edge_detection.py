from pathlib import Path
from tkinter.filedialog import askopenfilename, askdirectory
import hyperspy.api as hs
import numpy as np
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from tifffile import imwrite


def _vector_gradient(vecfield: np.ndarray, idx: tuple[int, int])\
        -> tuple[tuple[float, float], tuple[float, float]] | tuple[tuple[None, None], tuple[None, None]]:
    """Compute the vector gradient at a point in a 2D array.
    Args:
        vecfield: The entire vector field.
        idx: The 2D index of the point at which we want to compute the vector gradient.

    Returns: The local vector gradient tensor, in the form described at:
    https://math.stackexchange.com/questions/156880/what-does-it-mean-to-take-the-gradient-of-a-vector-field
    That is, for direction vectors (a, b): ((da/dx, da/dy),
                                            (db/dx, db/dy))
    For entries on the edge of the array, returns None.
    """
    this_x, this_y = idx[1], idx[0]
    ylim, xlim, _ = vecfield.shape

    # Edge check
    if this_x == 0 or this_y == 0 or this_x == xlim-1 or this_y == ylim-1:
        return ((None, None),
                (None, None))

    uppx = vecfield[this_y-1, this_x]
    dnpx = vecfield[this_y+1, this_x]
    lfpx = vecfield[this_y, this_x-1]
    rtpx = vecfield[this_y, this_x+1]

    dadx = float((lfpx[0] - rtpx[0]) / 2)
    dbdx = float((lfpx[1] - rtpx[1]) / 2)
    dady = float((uppx[0] - dnpx[0]) / 2)
    dbdy = float((uppx[1] - dnpx[1]) / 2)

    return ((dadx, dady),
            (dbdx, dbdy))


def _minmax_norm(img):
    _min, _max = np.min(img), np.max(img)
    normed = (img - _min) / (_max - _min)
    return normed


# %%
infile = hs.load(askopenfilename())

# %%
dpc_img = (next(signal for signal in infile if signal.metadata.General.title == "DPC"))
complex_data = dpc_img.data
phase_data = np.array([[np.angle(entry) for entry in row] for row in complex_data])

plt.imshow(phase_data, cmap="hsv")  # Sanity check for phase
plt.colorbar()

# %%
vector_data = np.array([[(np.cos(np.angle(entry)),
                          np.sin(np.angle(entry))) for entry in row] for row in complex_data])

tensor_data = np.array([_vector_gradient(vector_data, (i, j))
                        for i in np.arange(0, vector_data.shape[0])
                        for j in np.arange(0, vector_data.shape[1])]).reshape(vector_data.shape[0],
                                                                              vector_data.shape[1], 2, 2)

# %%
# Either all four entries in the tensor are None, or they all have floats
trimmed = tensor_data[tensor_data != ((None, None),
                                      (None, None))].reshape(tensor_data.shape[0]-2,  # Two borders to remove
                                                             tensor_data.shape[1]-2,  # in each direction
                                                             tensor_data.shape[2],
                                                             tensor_data.shape[3]).astype(np.float32)

# %%
divergences = np.array([tensor[0, 0] + tensor[1, 1] for row in trimmed for tensor in row]).reshape(trimmed.shape[0],
                                                                                                   trimmed.shape[1])

# %%
lowthresh = np.mean(divergences) - 0.5*np.std(divergences)
highthresh = np.mean(divergences) + 0.5*np.std(divergences)

plt.imshow(divergences, cmap="inferno", vmin=lowthresh, vmax=highthresh)
plt.colorbar()

# %%
directory = Path(askdirectory())
fname = infile[0].tmp_parameters.original_filename
imwrite(directory / (fname + "_deltax.tif"),
        (trimmed[:, :, 0, 0]+trimmed[:, :, 1, 0]).astype(np.float32),
        photometric="minisblack")
imwrite(directory / (fname + "_deltay.tif"),
        (trimmed[:, :, 0, 1]+trimmed[:, :, 1, 1]).astype(np.float32),
        photometric="minisblack")
imwrite(directory / (fname + "_divergence.tif"),
        (trimmed[:, :, 0, 0]+trimmed[:, :, 1, 1]).astype(np.float32),
        photometric="minisblack")
imwrite(directory / (fname + "_sumall.tif"),
        (trimmed[:, :, 0, 0]+trimmed[:, :, 0, 1]+trimmed[:, :, 1, 0]+trimmed[:, :, 1, 1]).astype(np.float32),
        photometric="minisblack")

# %%
# TODO: This is the start of an attempt to detect divergence edges via flow field (edge-tangent flow), still in progress
normed = _minmax_norm(divergences)
flow_field = np.zeros(normed.shape+(2,), dtype=np.float32)
sobel_x = ndimage.sobel(divergences, 0)
sobel_y = ndimage.sobel(divergences, 1)
sobel_mag = _minmax_norm(np.sqrt(sobel_x**2 + sobel_y**2))

for i in range(divergences.shape[0]):
    for j in range(divergences.shape[1]):
        a, b = sobel_x[i][j], sobel_y[i][j]
        flow_field[i][j] = np.tan(np.array([a, b]) / np.linalg.norm(np.array([a, b])))

#%%
