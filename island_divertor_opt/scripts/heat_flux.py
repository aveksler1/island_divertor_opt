from flare.mmesh import tasks
import sys

path = sys.argv[1]

tasks.init(
    path, 
    "boundary.nc", 
    "tracing.nc")
density = 1.E19
temperature = 50 #in eV
chi = 0.1 #in m^2/s
nparticles = 100000
tasks.heat_load_proxy(density, temperature, chi, nparticles,output="heat_flux.nc")