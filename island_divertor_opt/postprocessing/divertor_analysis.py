import numpy as np
import sys
import os
import subprocess


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

from moose.data import Dataset
from island_divertor_opt.postprocessing.heat_flux_diagnostic import HeatFluxDiagnostic
from island_divertor_opt.optimization import flare_postprocessing

path = sys.argv[1]

cmap = cm.magma
cmap.set_bad('white')
diag = HeatFluxDiagnostic(os.path.join(path, 'output'), verbose=True)

# Print out heat flux and hit percentages on every surface.
diag.calculate_percentage()
diag.calculate_peak_hflux()

# Plotting divertor heat flux
diag.plot(save=True)
diag.plot_sp(save=True)

# Saving divertor vtk files
subprocess.run(
    ["python", "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/postprocessing/div_to_vtk.py", os.path.join(path, "output"), "heat_flux.nc", "hload"]
)

