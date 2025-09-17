import os
import vtk
import numpy as np
import matplotlib.pyplot as plt
from moose.data import Dataset
import argparse

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("output_file", 
    help="Path to output file. Will be one of heat_flux.nc or strike_points.nc",
    type=str,
)
parser.add_argument("data_key", 
    help="Key for output file. Will be `hload` for heat_flux.nc and `p` for \
        strike_points.nc",
    type=str,
)
args = parser.parse_args()
output_file = args.output_file
data_key = args.data_key

print("Loading in data...", flush = True)

data = Dataset.loadnc(output_file)
p = data[data_key] 

data = p.plot(plot3d='vtk', vtk_show=None)

for i, g in enumerate(data):
    print(f"Saving boundary {i}", flush = True)

    writer = vtk.vtkXMLStructuredGridWriter()

    # Set the filename
    writer.SetFileName(os.path.join(project_path, f"boundary_{i}.vtp"))

    # Set the input VTK object
    writer.SetInputData(g)

    # Write the file
    writer.Write()