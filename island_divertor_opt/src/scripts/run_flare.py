from moose.geometry import Torosurf, Hypersurf3d
from flare.mmesh import tasks
from mpi4py import MPI
import sys
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

path = sys.argv[1]
os.chdir(path)

if rank == 0:
    target0 = Torosurf.loadtxt("../boundary/wall.txt", units="cm")
    target1 = Torosurf.loadtxt("../boundary/classical_fwd.txt", units="cm")
    target2 = Torosurf.loadtxt("../boundary/classical_bwd.txt", units="cm")
    target3 = Torosurf.loadtxt("../boundary/classical_fwd_cap.txt", units="cm")
    target4 = Torosurf.loadtxt("../boundary/classical_bwd_cap.txt", units="cm")
    
    boundary = Hypersurf3d({},
        {"wall": target0, "classical_fwd": target1, "classical_bwd": target2, 
        "classical_fwd_cap": target3, "classical_bwd_cap": target4}
    )
    
    boundary.savenc(os.path.join(path, "boundary.nc"))

comm.barrier()

tasks.init("/global/homes/a/aveksler/t1e/RUNS/input_files/grid/mmesh.nc", "boundary.nc", "tracing.nc")

tasks.strike_point_density(1e-6, 100000, output="strike_points.nc")

density = 1.E19
temperature = 50 #in eV
chi = 0.1 #in m^2/s
nparticles = 100000
tasks.heat_load_proxy(density, temperature, chi, nparticles, output="heat_flux.nc")