# Read in an array of TIF images, and stitch them together for simultaneous analysis

import os
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt

INPATH = r"C:\Users\charles\Documents\GitHub\abtem_scripts\_output\BZT_meas_pickle\thickness_series_haadfs"
REPS = 4
OUTPATH = r"C:\Users\charles\Documents"
OUTNAME = "BZT40 Thickness Gradient.tif"


def list_files(directory):
    files = []
    for file in os.listdir(directory):
        if file[-4:] == '.tif':
            files += [file]
        else:
            continue
    return(sorted(files))


def initialize_canvas(directory, files, reps):
    img = tif.imread(os.path.join(directory, files[0]))
    img_y, img_x = np.shape(img)

    if len(files) / reps % 1 != 0:
        raise RuntimeError("Number of images in directory is not a multiple of REPS")

    cvs_y = img_y * int(len(files) / reps)
    cvs_x = img_x * reps
    return np.zeros((cvs_x, cvs_y))


def stitch_images(directory, r):
    files = list_files(directory)
    canvas = initialize_canvas(directory, files, r)

    for i, file in enumerate(files):
        img = tif.imread(os.path.join(directory, file))
        y_size, x_size = np.shape(img)
        x = i//r + 1
        y = (i % r) + 1

        a = x_size * (x-1)
        b = a + x_size
        c = y_size * (y-1)
        d = c + y_size

        canvas[c:d, a:b] = img

    return canvas


image = (stitch_images(INPATH, REPS)*(2**16-1)).astype(np.uint16)
# %% Check to make sure the image looks right
plt.imshow(image)
# %% Save the image
tif.imwrite(os.path.join(OUTPATH, OUTNAME), image, photometric="minisblack", dtype="uint16")
