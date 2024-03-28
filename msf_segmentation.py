from pathlib import Path
from tkinter.filedialog import askopenfilename, askdirectory
import hyperspy.api as hs
import numpy as np
import scipy.ndimage as ndimage
from matplotlib import (pyplot as plt)
from tifffile import imwrite
import sklearn
from timeit import default_timer as timer

# This didn't work very well; noise issue?  Lattice being in the image also seems to be a confounding factor

# %%
infile = hs.load(askopenfilename())

# %%
dpc_img = (next(signal for signal in infile if signal.metadata.General.title == "DPC"))
complex_data = dpc_img.data
phase_data = np.array([[np.angle(entry) for entry in row] for row in complex_data])

plt.imshow(phase_data, cmap="hsv")  # Sanity check for phase
plt.colorbar()

# %%
# We need to downsample the image to make the MeanShift fast enough
# Blur first (even though it's marginally slower) to suppress lattice artifacts
blurred = ndimage.gaussian_filter(phase_data, sigma=10, mode="nearest")
downsampled = blurred[::4, ::4]
plt.imshow(downsampled, cmap="hsv")

# %%
# 2048x2048 takes ~29 hours, 1024x1024 takes between 1.5 - 2 hours
start = timer()
cluster_labels = sklearn.cluster.MeanShift(bandwidth=2, n_jobs=48).fit_predict(downsampled.flatten().reshape(-1, 1))
end = timer()
print(end-start)
#%%
cl_shaped = cluster_labels.reshape(downsampled.shape)
plt.imshow(cl_shaped, cmap="tab20")

#%%
