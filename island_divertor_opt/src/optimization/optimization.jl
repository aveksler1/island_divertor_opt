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
include("classical_divertor.jl")

HEAT_FLUX_WEIGHT = 1
CAP_WEIGHT = 1

DATABASE_PATH = "/global/u2/a/aveksler/DATABASE/flare"

# --- Read in DATABASE name
DATABASE_NAME = ARGS[1]

function build_divertor()
    """
    A function to build the divertor given the starting points and the strike 
    angle. 

    Builds JSON file, uses classical_divertor_improved to build divertor, and 
    passes data to FLARE. 
    """
    # Build divertor from json files

    json_input_files = ["closed_classical_divertor_fwd.json", "closed_classical_divertor_bwd.json"]

    for f in json_input_files
        make_classical_divertor_from_json(f; verbose=false)
    end
    
    path = joinpath(DATABASE_PATH, DATABASE_NAME)
    run(`mv closed_classical_fwd.txt closed_classical_bwd.txt closed_classical_fwd_cap.txt closed_classical_bwd_cap.txt $path`)
    run(`cp /global/homes/a/aveksler/DATABASE/flare/input_files/.boundary /global/homes/a/aveksler/DATABASE/flare/input_files/mmesh.nc  /global/homes/a/aveksler/DATABASE/flare/input_files/wall.txt $path`)
end

function get_island_spline(json_data, island_boundary, o_points)
    """
    Function that gets the island boundary and returns splines that can be used to generate starting point for the divertor based
    on parameters 0 - 2*pi.
    """
    points_per_period = json_data.points_per_period
    island = json_data.island
    n_islands = json_data.n_islands
    direction = json_data.direction
    eval_direction = json_data.eval_direction
    startϕdeg = json_data.start_phi_deg
    ix = Int(startϕdeg*2) + 1

    # Getting the spline of the island boundary
    risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
    # (r, z) of O-point
    ro = o_points[ix, 1]
    zo = o_points[ix, 2]
    # makes a spline of the island boundary
    spl = make_island_spline(risl, zisl, ro, zo, eval_direction)
    return spl
end

mkpath(joinpath(DATABASE_PATH, DATABASE_NAME))

println("Made path ", joinpath(DATABASE_PATH, DATABASE_NAME))

# --- Initialize starting parameters
json_input_file = "closed_classical_divertor_fwd.json"
json_data = JSON3.read(json_input_file)
island_boundary = readdlm(json_data.island_boundary_file,',')
o_points = get_o_points(
    json_data.o_points_file, json_data.island, json_data.n_islands
)
spline = get_island_spline(json_data, island_boundary, o_points)

function run_simulation(left_pt, right_pt, spline)
    """
    From input parameters (left starting point, right starting point
    strike angle), builds a divertor and runs flare to evaluate divertor
    performance.
    """
    # First we load in the JSON file and modify the points
    read_only_json = JSON3.read(joinpath(DATABASE_NAME, "classical_divertor_fwd.json"))
    writeable_json = copy(read_only_json)
    r1, z1 = spline(right_pt)
    r2, z2 = spline(left_pt)
    writeable_json[:r1] = r1
    writeable_json[:z1] = z1
    writeable_json[:r2] = r2
    writeable_json[:z2] = z2
    # writeable_json[:r1] = right_pt[1]
    # writeable_json[:z1] = right_pt[1]
    # writeable_json[:r2] = left_pt[1]
    # writeable_json[:z2] = left_pt[1]
    open(joinpath(DATABASE_PATH, DATABASE_NAME, "classical_divertor_fwd.json"), "w") do io
        JSON3.write(io, writeable_json)
    end

    classical_fwd, classical_bwd = build_divertor()

    run(`mpirun -np 64 run_flare.py $DATABASE_NAME`)

    pymodule = pyimport("flare_postprocessing") 
    q, hit_percentage = pymodule.get_cost_vars()
    return q, hit_percentage
end

function f(params)
    """
    Passes parameters to run simulation and tries to minimize this functions
    
    (heat_flux - target)? 
    (strike_angle - target)?
    negative sign on spread to maximize the spread?
    """
    # Passes left and right point into run simulations
    q, cap_hits = run_simulation(params[1], params[2], spline) 
    return HEAT_FLUX_WEIGHTS*q^2 + CAP_WEIGHT*min(cap_hits^2, 0)
end


theta_li = 1.2*pi
theta_ri = (3/4)*pi

# Initial parameter vector
x0 = [theta_li, theta_ri]

lower = [0, 0]
upper = [2*pi, 2*pi]

function g(theta_l, theta_r)
    dtheta = pi/60
    grad_theta_l = (
        -f([theta_l + 2*dtheta, theta_r]) + 
        8*f([theta_l + dtheta, theta_r]) - 
        8*f([theta_l - dtheta, theta_r]) + 
        f([theta_l - 2*dtheta, theta_r])
    ) / (12*dtheta)
    grad_theta_r = (
        -f([theta_l, theta_r+2*dtheta]) + 
        8*f([theta_l,  theta_r+dtheta]) - 
        8*f([theta_l, theta_r - dtheta]) + 
        f([theta_l, theta_r - 2*dtheta])
    ) / (12*dtheta)
    return [grad_theta_l, grad_theta_r]
end

# Perform the optimization
# od = OnceDifferentiable(f, g; inplace=false)
# result = optimize(od, lower, upper, x0, Fminbox(LBFGS()); inplace=false)
res = optimize(f, g, lower, upper, x0, GradientDescent())
               # Optim.Options(g_tol = 1e-12,
               #               iterations = 10,
               #               store_trace = true,
               #               show_trace = false,
               #               show_warnings = true))


# num = 100
# r = zeros(num)
# z = zeros(num)
# thetas = range(0, stop=2*pi, length=num)
# for (i, theta) in enumerate(thetas)
#     _r, _z = spline(theta)
#     r[i] = _r
#     z[i] = _z
# end

# p = Plots.scatter(
#     r, z,
#     marker_z = thetas,
#     markercolor = :inferno,
#     colorbar = true,
# )
# savefig(p, "spline.png")
# println(spline(0))
# println(spline(pi/4))
# println(spline(pi/2))
# println(spline(pi))
# println(spline(2*pi))

# optimize(objective_function, params, LBFGS()) 