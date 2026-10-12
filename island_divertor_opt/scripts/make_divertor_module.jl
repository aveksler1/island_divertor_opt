module DivertorMaker

using MagneticFieldToolkit
using LinearAlgebra
using StaticArrays
using CoordinateTransformations
using Roots
using DelimitedFiles
using Dierckx
using JSON3
using Statistics
using Measures
using NaNStatistics
using FilePathsBase

# include("/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/classical_divertor_final.jl")
include("/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/classical_divertor_final2.jl")

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

function save_divertor(left_pt, right_pt, spline, json_input_files)
    """
    From input parameters (left starting point, right starting point
    strike angle), builds a divertor.
    """
    target_dir = joinpath(pwd(), "boundary")
    mkpath(target_dir)

    for f in json_input_files
        read_only_json = JSON3.read(f)
        writeable_json = copy(read_only_json)

        r1, z1 = spline(right_pt)
        r2, z2 = spline(left_pt)

        writeable_json[:r1] = r1
        writeable_json[:z1] = z1
        writeable_json[:r2] = r2
        writeable_json[:z2] = z2

        original_filename = basename(f)
        new_file_path = joinpath(target_dir, original_filename)

        open(new_file_path, "w") do io
            JSON3.write(io, writeable_json)
        end

        make_classical_divertor_from_json(new_file_path; verbose=false)
    end
end



function run_divertor_creation(left_pt::Float64, right_pt::Float64, fwd_file_name, bwd_file_name)

    println("Creating a divertor with (theta_l, theta_r) = ($left_pt, $right_pt)")

    # json_input_files = [
    #     "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/classical_divertor_fwd.json", 
    #     "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/classical_divertor_bwd.json"
    # ]

    base_path = "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/"
    json_input_files = [
        joinpath(base_path, fwd_file_name),
        joinpath(base_path, bwd_file_name)
    ]

    for file in json_input_files
        json_data = JSON3.read(file)
        island_boundary = readdlm(json_data.island_boundary_file, ',')
        o_points = get_o_points(
            json_data.o_points_file, json_data.island, json_data.n_islands
        )
        spline = get_island_spline(json_data, island_boundary, o_points)
        
        save_divertor(left_pt, right_pt, spline, json_input_files)
    end

end

end # module