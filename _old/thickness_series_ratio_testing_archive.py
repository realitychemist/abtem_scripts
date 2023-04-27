# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:59:43 2023

@author: charles
"""

# This was originally part of a larger script, it's saved here for archival purposes.  If it needs
#  to work again, it will probably take a bit of hacking together.

# Select the correct B sites (the ones centered in full B site columns)
# Then assign them their correct thicknesses

thickness_list = [1.6332, 2.4498, 2.8581, 3.2664, 3.6747, 4.4913, 4.8996, 5.3079, 5.7162, 6.5328,
                  6.9411, 7.3494, 7.7577, 8.5743, 8.9826, 9.3909, 9.7992, 10.6158, 11.0241,
                  11.4324, 11.8407, 12.6573, 13.0656, 13.4739, 13.8822, 14.6988, 15.1071, 15.5154,
                  15.9237, 16.7403, 17.1486, 17.5569, 17.9652, 18.7818, 19.1901, 19.5984, 20.0067,
                  20.8233, 21.6399, 22.8648, 23.6814, 24.9063, 25.7229, 26.9478]


def assign_sites(df, thickness_list, image_width=3, image_height=None, reps=4):
    if image_height is None:
        image_height = image_width
    df_new = deepcopy(df)
    # Min_u, max_v should be the top-leftmost B site
    df_new["selected"] = df_new.apply(lambda _: False, axis=1)
    df_new["est_thickness"] = df_new.apply(lambda _: False, axis=1)
    min_u, max_v = min(df_new["u"]), max(df_new["v"])
    initial_uvs = (min_u + 1, max_v - 1)
    for i, t in enumerate(thickness_list):
        u = initial_uvs[0] + i*image_width
        for j in range(reps):
            v = initial_uvs[1] - j*image_height
            df_new.loc[(df_new["u"] == u) & (df_new["v"] == v), "selected"] = True
            df_new.loc[(df_new["u"] == u) & (df_new["v"] == v), "est_thickness"] = t

    return df_new[df_new["selected"]]


bframe_subset = assign_sites(bframe, thickness_list)

ratio_dict = {}
stdev_dict = {}
for _, row in bframe_subset.iterrows():
    ratio_dict[row["est_thickness"]] = []
for _, row in bframe_subset.iterrows():
    ratio_dict[row["est_thickness"]].append(row["int_ratio"])
for key, values in ratio_dict.items():
    stdev_dict[key] = np.std(values)
    ratio_dict[key] = np.mean(values)

plt.plot(ratio_dict.keys(), ratio_dict.values(), "r-")
plt.fill_between(ratio_dict.keys(), [r-s for r, s in zip(ratio_dict.values(),
                                                          stdev_dict.values())],
                  [r+s for r, s in zip(ratio_dict.values(), stdev_dict.values())],
                  color='#ff000080')