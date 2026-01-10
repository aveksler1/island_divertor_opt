import os
import sys

# --- CONFIGURATION ---
JULIA_BIN = "/pscratch/sd/a/aveksler/julia/juliaup/julia-1.11.5+0.x64.linux.gnu/bin/julia"
JULIA_ENV = "/pscratch/sd/a/aveksler/julia/environments/v1.11/"
JULIA_DEPOT = "/pscratch/sd/a/aveksler/julia"

# --- THE HDF5 FIX (CRITICAL) ---
# We take the path from your error message and force the system to read this folder first.
# This ensures Julia finds its own HDF5, not the incompatible one from Conda.
hdf5_artifact_path = "/pscratch/sd/a/aveksler/julia/artifacts/2117b531439d782e1cb0ebffcc4dcf11d274cc39/lib"

def _setup_environment():
    # 1. Standard Julia variables
    os.environ["JULIA_DEPOT_PATH"] = JULIA_DEPOT
    os.environ["PYTHON_JULIACALL_EXE"] = JULIA_BIN
    os.environ["PYTHON_JULIACALL_PROJECT"] = JULIA_ENV
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    os.environ["JULIA_NUM_THREADS"] = "8"

    # 2. Library Path Construction
    #    Order matters: [HDF5 Artifact] -> [Julia System Libs] -> [Conda/System Libs]
    
    julia_lib_dir = os.path.join(os.path.dirname(os.path.dirname(JULIA_BIN)), "lib")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")

    # We build the new path with HDF5 FIRST
    new_ld_path = f"{hdf5_artifact_path}:{julia_lib_dir}:{current_ld}"
    
    # Apply it
    os.environ["LD_LIBRARY_PATH"] = new_ld_path

    # 3. Cleanup
    for key in ["JULIA_BINDIR", "JULIA_HOME"]:
        os.environ.pop(key, None)

_setup_environment()

# --- IMPORT ---
import juliacall
from juliacall import Main as jl

print(f"✅ Bridge Loaded.")