import os
import pandas
import matplotlib.pyplot as plt

path = r"C:\Users\charles\Downloads\will_eds"

#%%
fname = "1215 20220708 340 kx HAADF SI.csv"

with open(os.path.join(path, fname)) as infile:
    data = pandas.read_csv(infile)


fig, ax1 = plt.subplots(figsize=(32, 18), dpi=200)
ax2 = ax1.twinx()

ax1.set_xlabel("Position (nm)")

ax1.set_ylim(min(data["HAADF"])*0.95/1000, max(data["HAADF"])*1.05/1000)
ax1.set_ylabel("HAADF Intensity (kCounts)")

ax2.set_ylim(0, max(data["W"])*1.05)
ax2.set_ylabel("EDS Intensity (Counts)")

ax1.plot(data["Position"]*1e9, data["HAADF"]/1000, color="gray", label="HAADF")
ax2.plot(data["Position"]*1e9, data["W"], color="blue", label="W")
ax2.plot(data["Position"]*1e9, data["Pt"], color="red", label="Pt")

fig.legend(loc=(0.15, 0.7))

#%%

fname = "1215 20220708 340 kx HAADF SI spectra.csv"
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (32, 12)

with open(os.path.join(path, fname)) as infile:
    data = pandas.read_csv(infile)
    data.rename(columns={"Spectra from Area #1-Spectrum": "spec"}, inplace=True)

plt.xlim(0, 12.5)
plt.xlabel("Energy (keV)", fontsize="28")
plt.xticks(fontsize="xx-large")

plt.ylim(0, max(data["spec"])*1.075/1000)
plt.ylabel("Intensity (kCounts)", fontsize="28")
plt.yticks(fontsize="xx-large")

plt.fill_between(data["Energy"]/1000, data["spec"]/1000, antialiased=True, edgecolor="darkblue")

peak_energies = [0.3924, 0.5249, 1.4865, 1.7397, 2.0505, 8.3976, 9.4421, 9.6724, 9.9614]
peak_names = ["N", "O", "Al", "Si", "Pt-M", "W-Lα", "Pt-Lα", "W-Lβ1", "W-Lβ2"]
peak_ys = [0.5, 0.7, 4, 4.6, 0.5, 1.33, 0.5, 0.7, 0.4]

plt.vlines(peak_energies,
           0, peak_ys,
           color="black", linewidth=0.5)

for e, n, y in zip(peak_energies, peak_names, peak_ys):
    plt.text(e, y+0.05, n, fontsize="xx-large", ha="center")
    
plt.savefig(os.path.join(r"C:\Users\charles\Desktop", "Figure_2.png"))

# α β


