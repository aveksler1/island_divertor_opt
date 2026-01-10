import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

from moose.data import Dataset
from island_divertor_opt.postprocessing.heat_flux_diagnostic import HeatFluxDiagnostic
from island_divertor_opt.optimization import flare_postprocessing

cmap = cm.magma
cmap.set_bad('white')

path = sys.argv[1]

diag = HeatFluxDiagnostic(os.path.join(path, 'output'), verbose=True)

diag.plot(save=True)
diag.plot_sp(save=True)

# try:
#     diag.plot(save=True)
# except:
#     print("Error saving heat flux.")
# try:
#     diag.plot_sp(save=True)
# except:
#     print("Error saving strike point.")

# grid_fwd = diag.hflux['hload'].grid.xcart[1]
# data_fwd = diag.hflux['hload'].values[1]
# area_fwd = diag.hflux['area'].values[1]
# grid_bwd = diag.hflux['hload'].grid.xcart[2]
# data_bwd = diag.hflux['hload'].values[2]
# area_bwd = diag.hflux['area'].values[2]

# output_fwd, bins = diag.reformat_data(grid_fwd, data_fwd, area_fwd)
# output_bwd, bins = diag.reformat_data(grid_bwd, data_bwd, area_bwd)

# output = np.hstack((output_bwd, output_fwd))
# xax = np.linspace(15, 45, output_fwd.shape[1] + output_bwd.shape[1] + 1)

# fig, ax = plt.subplots(1, 1)

# im0 = ax.pcolormesh(bins, xax, output.T * diag.power, cmap=cmap, shading='flat')

# plt.colorbar(im0, ax=ax, label = "Heat Flux [$\\frac{MW}{m^2}$]")
# ax.set_xlabel("distance to center (m)")
# ax.set_ylabel("$\\phi$ (deg.)")
# fig.tight_layout()
# plt.savefig("heat_flux.png")

