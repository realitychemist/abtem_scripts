# Analysis Imports
import os
import tifffile
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import KDTree
from scipy.signal import hilbert
# Plotting Imports
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from skimage import color
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

# %% LOAD EDS MAPS

fpath = r"E:\Users\Charles\DMREF 20230306-1 C5-1543 SI"
fname_per_element = {"In": "DMREF 20230306-1 C5-1534-In-int.tif",
                     "Ga": "DMREF 20230306-1 C5-1534-Ga-int.tif",
                     "N":  "DMREF 20230306-1 C5-1534-N-int.tif",
                     "Al": "DMREF 20230306-1 C5-1534-Al-int.tif",
                     "Sc": "DMREF 20230306-1 C5-1534-Sc-int.tif"}
fname_haadf = "DMREF 20230306-1 C5-1534-HAADF.tif"

simg = {}

for elem in fname_per_element:
    with open(os.path.join(fpath, fname_per_element[elem]), "rb") as si_file:
        simg[elem] = tifffile.imread(si_file)
with open(os.path.join(fpath, fname_haadf), "rb") as haadf_file:
    himg = tifffile.imread(haadf_file)

# Clean up the namespace
del elem, fpath, fname_per_element, fname_haadf, si_file, haadf_file

# %% PLOT COLORIZED EDS MAP

b = simg["N"].astype(bool).astype(float)
g = simg["Al"].astype(bool).astype(float)
r = simg["Sc"].astype(bool).astype(float)

fig = plt.figure()
ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
rgb, _, _, _ = ax.imshow_rgb(r, g, b, interpolation="none")  # Make the plot, save the RGB axes

# Now we need to make the black parts of the RGB image transparent
rgb = rgb._A
rs, gs, bs = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
a = np.logical_or(np.logical_or(rs, gs), bs).astype(float)
rgba = np.dstack([r, g, b, a])
ax.RGB.clear()  # Remove the old RGB plot from the axes (clearing the non-transparent black)
ax.RGB.imshow(rgba, zorder=0, interpolation="none")
ax.RGB.imshow(himg, cmap="binary_r", zorder=-1)  # Put the HAADF behind the RGB image

ax.RGB.set_title("Al$_{0.7}$Sc$_{0.3}$N EDS Map")
ax.R.set_title("Scandium", color="white", y=0.9)
ax.G.set_title("Aluminum", color="white", y=0.9)
ax.B.set_title("Nitrogen", color="white", y=0.9)
ax.RGB.tick_params(direction="in", color="white", left=False, bottom=False,
                   labelleft=False, labelbottom=False)
ax.R.tick_params(left=False)
ax.G.tick_params(left=False)
ax.B.tick_params(left=False)

# Clean up the namespace
del fig, ax, r, g, b, rs, gs, bs, a, rgb, rgba

# %% STATISTICS -- RDFs

rs = np.linspace(0, 10, num=50)
_rs = rs[1:]
dr = rs[1] - rs[0]  # Done this way in case we ever don't start from 0
shell_areas = [np.pi * (_rs[i]**2 - rs[i]**2) for i in range(len(_rs))]
gs = {}

for elem in ["Al", "Sc", "N"]:
    points = list(zip(*np.where(simg[elem])))
    density = len(points) / np.prod(simg[elem].shape)
    weights = [simg[elem][pt] for pt in points]
    tree = KDTree(points)
    ns = tree.count_neighbors(tree, rs, weights=weights)
    ns -= ns[0]  # Remove self-counting
    dns = [int(ns[i] - ns[i-1]) for i in range(1, len(ns))]  # Per-shell instead of cumulative
    g = [dn / (a * density) for dn, a in zip(dns, shell_areas)]
    gs[elem] = g

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(_rs, gs["Al"], color="green")
axs[1].plot(_rs, gs["Sc"], color="red")
axs[2].plot(_rs, gs["N"], color="blue")

# %% STATISTICS -- AUTOCORRELATION

def norm_auto_correlation(image):
    """
    Calculates normalized autocorrelation of an image
    """
    im_pad = image - np.mean(image)
    pad = int(np.min(image.shape)/2)
    im_pad = np.pad(
        im_pad,
        pad_width=pad,
        mode='constant'
        )
    
    im_norm = np.pad(
        np.ones_like(image), 
        pad_width=pad,
        mode='constant'
        )
    
    im_nac = np.real(
        np.fft.ifft2(np.abs(np.fft.fft2(im_pad))**2)
        ) / np.sum(im_pad**2)
    
    im_norm = np.real(
        np.fft.ifft2(np.abs(np.fft.fft2(im_norm))**2)
        )  / np.sum(im_norm**2)
    
    sub = im_norm > 0
    
    im_nac[sub] /= im_norm[sub]
    
    return im_nac

acs = {}
for elem in ["Al", "Sc", "N"]:
    acs[elem] = norm_auto_correlation(simg[elem])

plt.imshow(acs["N"], cmap="binary")