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
include("classical_divertor_l-_r+.jl")

json_input_files = [
    "classical_divertor_fwd.json", 
    "classical_divertor_bwd.json"
]


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

function save_divertor(left_pt, right_pt, spline)
    """
    From input parameters (left starting point, right starting point
    strike angle), builds a divertor.
    """
    # First we load in the JSON file and modify the points

    for f in json_input_files
        read_only_json = JSON3.read(f)
        writeable_json = copy(read_only_json)
        r1, z1 = spline(right_pt)
        r2, z2 = spline(left_pt)
        writeable_json[:r1] = r1
        writeable_json[:z1] = z1
        writeable_json[:r2] = r2
        writeable_json[:z2] = z2
        open(f, "w") do io
            JSON3.write(io, writeable_json)
        end
    end

    for f in json_input_files
        make_classical_divertor_from_json(f; verbose=false)
    end
end

if length(ARGS) != 2
    error("Requires 2 angles")
end

# The ARGS array contains the arguments as strings.
arg1_str = ARGS[1]
arg2_str = ARGS[2]

# Convert the strings to integers.
theta_l = parse(Float64, arg1_str)
theta_r = parse(Float64, arg2_str)

println("Creating a divertor with (theta_l, theta_r) = ($arg1_str, $arg2_str)")

for file in json_input_files
    json_data = JSON3.read(file)
    island_boundary = readdlm(json_data.island_boundary_file,',')
    o_points = get_o_points(
        json_data.o_points_file, json_data.island, json_data.n_islands
    )
    spline = get_island_spline(json_data, island_boundary, o_points)
    save_divertor(theta_l, theta_r, spline)
end