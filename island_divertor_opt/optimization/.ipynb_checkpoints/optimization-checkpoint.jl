using MagneticFieldToolkit
using LinearAlgebra
using StaticArrays
using CoordinateTransformations
using Roots
using DelimitedFiles
using Dierckx
using JSON3
using Plots
using Statistics
using Measures
using NaNStatistics
using Optim
using PyCall
using Optimization
using OptimizationBBO

# --- Define iteration count
iter = 0 
root_dir = pwd()

function run_simulation(left_pt, right_pt)
    """
    From input parameters (left starting point, right starting point
    strike angle), builds a divertor and runs flare to evaluate divertor
    performance.
    """
    global iter
    # global root_dir
    # cd(root_dir)
    println("Setting up environment.")
    # Setup folders inside of opt run
    dirname = "opt_$(iter)"
    run(`mkdir $dirname`)
    cd("$dirname")
    run(`mkdir boundary`)
    run(`mkdir output`)

    run(`cp /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/classical_divertor_fwd.json .`)
    run(`cp /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/classical_divertor_bwd.json .`)
    
    println("Great Success!")
    
    println("Running divertor.")

    # Make the actual divertor given the left point and right point
    run(`julia /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/make_divertor_final.jl $left_pt $right_pt`)
    println("Great Success!")
    # Move the divertor into the boundary file
    println("Moving files.")
    run(`mv classical_fwd.txt classical_bwd.txt classical_fwd_cap.txt classical_bwd_cap.txt boundary`)
    run(`cp /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/wall.txt boundary`)
    println("Great Success!")
    
    # Enter the output folder
    cd("output")
    
    # Run flare in the output folder
    println("Running flare.")
    run(`echo $(pwd())`)

    run(`mpirun -np 32 python /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/run_flare.py $(pwd())`)
    println("Great Success!")
    
    # Performing postprocessing
    cd("../..")
    pymodule = pyimport("/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/optimization/flare_postprocessing") 
    dirname = "opt_$(iter)/output"

    # Get cost vars 
    println("Running postprocessing...")
    qpeak, qtot, cap_hits = pymodule.get_cost_vars(dirname)
    println("Great Success!")

    return qpeak, qtot, cap_hits
end

function cost(u, p)
    """
    Passes parameters to run simulation and tries to minimize this functions
    """
    global iter
    # Passes left and right point into run simulations
    println("Running simulation on iter $iter")
    lp = u[1]
    rp = u[2]
    println("Left point: $lp; Right point: $rp")
    flush(stdout)
    qpeak, qtot, cap_hits = run_simulation(u[1], u[2]) 
    # Compute cost
    money = pymodule.calculate_cost(qpeak, qtot, cap_hits)
    println("Iter $iter: Cost = $money")
    println("Great Success!")
    # Increase iteration count
    iter += 1

    return money
end

# Initial parameter vector and bounds
theta_li = 4.1
theta_ri = 1.3
u0 = [theta_li, theta_ri]
lower = [(3/2)*pi, 0]
upper = [2*pi, (3/2)*pi]

# Perform the optimization
# f = OptimizationFunction(cost)
prob = Optimization.OptimizationProblem(cost, u0, lb = lower, ub = upper)
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=10)