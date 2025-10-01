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
prev_cost = 0
flare_post = pyimport("island_divertor_opt.optimization.flare_postprocessing")

println("Root Directory: $(root_dir)")

function run_simulation(left_pt, right_pt)
    """
    From input parameters (left starting point, right starting point
    strike angle), builds a divertor and runs flare to evaluate divertor
    performance.
    """
    global iter
    global root_dir
    global flare_post

    println("Setting up environment.")
    cd(root_dir)
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
    run(`julia /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/make_divertor_final.jl $left_pt $right_pt`)
    println("Great Success!")
    # Move the divertor into the boundary file
    println("Moving files.")
    run(`mv classical_fwd.txt classical_bwd.txt classical_fwd_cap.txt classical_bwd_cap.txt boundary`)
    run(`cp /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/wall.txt boundary`)
    println("Great Success!")
    
    # Enter the output folder
    cd("output")
    
    # Run flare in the output folder
    println("Running flare.")

    run(`mpirun -np 64 python /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/run_flare.py $(pwd())`)
    println("Great Success!")
    
    # # Performing postprocessing
    dirname = "$root_dir/opt_$(iter)/output"

    # Get cost vars 
    println("Running postprocessing on dir $(dirname)...")
    qpeak, qtot, cap_hits = flare_post.get_cost_vars(dirname)
    println("Great Success!")

    return qpeak, qtot, cap_hits
end

function cost(u, p)
    """
    Passes parameters to run simulation and tries to minimize this functions
    """
    global iter
    global flare_post
    global prev_cost
    # Passes left and right point into run simulations
    println("=================================================================")
    lp = u[1]
    rp = u[2]
    println("Running sim on iter $iter; (left point, right point = ($lp, $rp))")
    qpeak, qtot, cap_hits = run_simulation(u[1], u[2]) 
    # Compute cost
    money = flare_post.calculate_cost(qpeak, qtot, cap_hits)
    println("Iter $iter: Cost = $money")
    if iter != 1
        println("Iter $iter: Cost = $money")
        pchange = ((money - prev_cost)/prev_cost)*100
        println("% cost change: $pchange")
    else 
        println("Iter $iter: Cost = $money")
        prev_cost = money
    end
    flush(stdout)
    # Increase iteration count
    iter += 1

    return money
end

# Initial parameter vector and bounds
theta_li = 4.1
theta_ri = 1.3
u0 = [theta_li, theta_ri]
lower = [pi, 0]
upper = [2*pi, pi]

# Perform the optimization
# f = OptimizationFunction(cost)
prob = Optimization.OptimizationProblem(cost, u0, lb = lower, ub = upper)
sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=100)
