import sys
import os
import argparse
from mpi4py import MPI

from moose.geometry import Torosurf, Hypersurf3d

from flare.mmesh import tasks
from flare.mmesh.unstructured import SolidSurfaces

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("path",
    help= "Path to change into.",
    type=str,
)
parser.add_argument("--chi", 
    help="chi",
    type=float,
)
parser.add_argument("--bstep", 
    help="Stepsize in magnetic field direction",
    default=0.05,
    type=float,
)
args = parser.parse_args()
path = args.path
chi=args.chi
bstep = args.bstep

os.chdir(path)

if rank == 0:
    target0 = Torosurf.loadtxt("../boundary/wall.txt", units="cm")
    target1 = Torosurf.loadtxt("../boundary/classical_fwd.txt", units="cm")
    target2 = Torosurf.loadtxt("../boundary/classical_bwd.txt", units="cm")
    target3 = Torosurf.loadtxt("../boundary/classical_fwd_cap.txt", units="cm")
    target4 = Torosurf.loadtxt("../boundary/classical_bwd_cap.txt", units="cm")
    
    boundary = SolidSurfaces(
        {"wall": target0, "classical_fwd": target1, "classical_bwd": target2, 
        "classical_fwd_cap": target3, "classical_bwd_cap": target4}
    )
    
    boundary.savenc(os.path.join(path, "boundary.nc"))

comm.barrier()

tasks.init_workspace("/global/homes/a/aveksler/t1e/RUNS/input_files/grid/mmesh.nc")
tasks.set_boundary("boundary.nc")


# tasks.strike_point_density(diffusion_coeff, 100000, output="strike_points.nc", bstep=bstep)

density = 1.E19
temperature = 50 #in eV
nparticles = 100000
tasks.heat_load_proxy(density, temperature, chi, nparticles, output="heat_flux.nc")