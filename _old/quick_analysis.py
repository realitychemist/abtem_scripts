# ASE Imports
from ase.io import read
# AbTEM Imports
# from abtem.potentials import Potential
# from abtem.waves import SMatrix
# from abtem.scan import GridScan
# from abtem.detect import AnnularDetector
# from abtem.detect import SegmentedDetector
# from abtem.temperature import FrozenPhonons
# Other Imports
import os
# import tifffile
import matplotlib.pyplot as plt
import numpy as np
from SingleOrigin.SingleOrigin import utils as so
from sklearn.neighbors import KernelDensity
from scipy.spatial import KDTree
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from skimage import color
import cmasher

# %%


def distribution_by_periodic_KDTree(atoms, k, nbins, bwidth):
    """Get a radial distribution function g(r) for a sublattice.

    A KDTree with periodic boundary conditions is used for fast queries which take into account the
    case where nearest neighbors are across the periodic boundary.  Building and querying the tree
    is quite fast.  Smoothing via gaussian kernel density estimate is also performed.

    Parameters
    ----------
    sublat : DataFrame
        A DataFrame containing the positions of the atoms from the model
    k : int
        The maximum number of nearest neighbors to count.  Increasing k will increase the maximum
        distance to which the RDF can be plotted, but will also slow down the calculations.  If set
        to a very large value, not all neighbors may be counted (only neighbors out to half the box
        width can be counted to avoid double-counting of neighbors in the periodic image).
    nbins : int
        The number of bins used to subsample the neighbor distances.  Increasing nbins will
        increase the accuracy of the gaussian kernel density estimation, but will dramatically
        slow down the calculations.
    bwide : float
        The bandwidth used for the gaussian kernel density estimation.  This is a hyperparameter
        which may need to be tuned for each particular case to give good results.  Higher values
        will increase smoothing.  This also affects the speed of calculations, but it is not
        recommended to change this parameter for the purpose of speeding up calculations.  Instead,
        change k or nbins.

    Returns
    -------
    g : list of floats
        The density as a function of radius, as derived from a kernel density estimation
    r : list of floats
        The radii corresponding to each of the densities returned in g.  Radius ranges from
        0 to the ceiling of the highest value found when computing neighbor distances
    """
    print("Constructing query tree...")
    kdtree = KDTree(atoms.positions)

    print("Done!  Finding neighbor distances...")
    d, _ = kdtree.query(atoms.positions, k=k)  # k is a hyperparameter
    d = np.delete(d, 0, 1)  # d includes a column of zeroes that need to be removed
    d = d.ravel(order="K")  # Flatten d to 1D; K should be fastest, does not preserve ordering

    print("Done!  Estimating kernel density...")
    # Bandwidth is a hyperparameter; might be a good idea to design an estimator for it
    kde = KernelDensity(kernel="gaussian", bandwidth=bwidth).fit(d.reshape(-1, 1))
    log_g = kde.score_samples(np.linspace(0, max(d), nbins).reshape(-1, 1))
    g = np.exp(log_g)
    r = np.linspace(0, max(d), nbins)
    print("Done!")

    return g, r


# %%

fpath = r"C:\Users\charles\Documents\AlScN\raw"

fname_0 = "AlN.xyz"
atoms_0 = read(os.path.join(fpath, fname_0))

fname_5 = "0.055.xyz"
atoms_5 = read(os.path.join(fpath, fname_5))

fname_50 = "0.5.xyz"
atoms_50 = read(os.path.join(fpath, fname_50))

# %%

g_5, r_5 = distribution_by_periodic_KDTree(atoms_5, 50, 10000, 0.01)
g_50, r_50 = distribution_by_periodic_KDTree(atoms_50, 50, 10000, 0.01)

# %%

perfect = set([round(np.linalg.norm(coord1 - coord2), 3) for coord1 in atoms_0.positions
               for coord2 in atoms_0.positions]) - {0}
nns = sorted(list(perfect))[:13]
# %%

plt.fill_between(r_50, g_50, color="#785EF0", label="(Al$_{0.5}$Sc$_{0.5}$)N")
plt.fill_between(r_5, g_5, color="#FE6100", alpha=0.66, label="(Al$_{0.945}$Sc$_{0.055}$)N")
plt.vlines(nns, 0, 2, color="#000000", linestyle="dotted", label="AlN Perfect Structure")
plt.title("Radial Distribution for (Al$_{1-x}$Sc$_x$)N")
plt.xlabel("Radius [A]")
plt.ylabel("g(r)")
plt.xticks(np.arange(0, max(r_5)+1, 1))
plt.xlim([1.5, 5.5])
plt.ylim([0, max(g_5)+0.1*max(g_5)])
plt.tick_params(axis='y', which='both', left=False, labelleft=False)
plt.legend(loc="upper left")
plt.show()

# %%

n_coords = atoms_5.positions[atoms_50.numbers == 7]
al_coords = atoms_5.positions[atoms_50.numbers == 13]
sc_coords = atoms_5.positions[atoms_50.numbers == 21]

n_n_vpcf = so.v_pcf(xlim=(-10, 10), ylim=(-10, 10), coords1=n_coords[:, 1:])
al_n_vpcf = so.v_pcf(xlim=(-10, 10), ylim=(-10, 10),
                     coords1=n_coords[:, 1:], coords2=al_coords[:, 1:])
sc_n_vpcf = so.v_pcf(xlim=(-10, 10), ylim=(-10, 10),
                     coords1=n_coords[:, 1:], coords2=sc_coords[:, 1:])

origin_x, origin_y = n_n_vpcf[1][0], n_n_vpcf[1][1]

norm_n = Normalize(vmin=0, vmax=max(n_n_vpcf[0].flatten()), clip=True)
mapper_n = cm.ScalarMappable(norm=norm_n, cmap=plt.get_cmap("cmr.arctic"))
norm_al = Normalize(vmin=0, vmax=max(al_n_vpcf[0].flatten()), clip=True)
mapper_al = cm.ScalarMappable(norm=norm_al, cmap=plt.get_cmap("cmr.nuclear"))
norm_sc = Normalize(vmin=0, vmax=max(sc_n_vpcf[0].flatten()), clip=True)
mapper_sc = cm.ScalarMappable(norm=norm_sc, cmap=plt.get_cmap("cmr.amber"))

n_rgb_array = mapper_n.to_rgba(n_n_vpcf[0])[:, :, :3]

n_rgb_array[180:210, 180:210] = 0  # Patch out down-column information that shouldn't be visible

al_rgb_array = mapper_al.to_rgba(al_n_vpcf[0])[:, :, :3]
sc_rgb_array = mapper_sc.to_rgba(sc_n_vpcf[0])[:, :, :3]

n_lab_array = color.rgb2lab(n_rgb_array)
al_lab_array = color.rgb2lab(al_rgb_array)
sc_lab_array = color.rgb2lab(sc_rgb_array)

n_l, al_l, sc_l = n_lab_array[:, :, 0], al_lab_array[:, :, 0], sc_lab_array[:, :, 0]
n_a, al_a, sc_a = n_lab_array[:, :, 1], al_lab_array[:, :, 1], sc_lab_array[:, :, 1]
n_b, al_b, sc_b = n_lab_array[:, :, 2], al_lab_array[:, :, 2], sc_lab_array[:, :, 2]

summed_l = n_l + al_l + sc_l
summed_l = summed_l / max(summed_l.flatten())*100

avg_a, avg_b = np.mean([n_a, al_a, sc_a], axis=0), np.mean([n_b, al_b, sc_b], axis=0)
merged_lab_array = np.dstack((summed_l, avg_a, avg_b))

merged_rgb_array = color.lab2rgb(merged_lab_array)
merged_hsv_array = color.rgb2hsv(merged_rgb_array)
hues, sats, values = (merged_hsv_array[:, :, 0],
                      merged_hsv_array[:, :, 1],
                      merged_hsv_array[:, :, 2])
sats = sats / max(sats.flatten())
merged_hsv_array = np.dstack((hues, sats, values))
merged_rgb_array = color.hsv2rgb(merged_hsv_array)

plt.imshow(merged_rgb_array)

plt.scatter(origin_x, origin_y, marker="+", color="#FFFFFF", s=100)
plt.title("Al$_{0.945}$Sc$_{0.055}$N (x | N)$_{110}$ vPCF", fontsize=16)
plt.tick_params(axis="x", bottom=False, labelbottom=False)
plt.tick_params(axis="y", left=False, labelleft=False)
plt.xlabel(r"[$\overline{1}$10]$\longrightarrow$", fontsize=12, loc="left")
plt.ylabel(r"[001]$\longrightarrow$", fontsize=12, loc="bottom")

legend_elements = [Patch(facecolor="blue", edgecolor="blue", label="(N | N)"),
                   Patch(facecolor="green", edgecolor="green", label="(Al | N)"),
                   Patch(facecolor="red", edgecolor="red", label="(Sc | N)")]
plt.legend(handles=legend_elements, loc="best")
plt.show()

#%%
