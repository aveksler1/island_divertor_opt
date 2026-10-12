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

include("/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/classical_divertor_final.jl")

# Cache to store loaded magnetic field objects in worker process memory
const MG_CACHE = Dict{String, Any}()

function get_cached_mg(mgrid_file::String)
    clean_path = strip(mgrid_file, ['"', '\'', ' '])
    if !haskey(MG_CACHE, clean_path)
        if endswith(clean_path, ".h5")
            MG_CACHE[clean_path] = read_hint_field(clean_path)
        else
            MG_CACHE[clean_path] = read_mgrid(clean_path, [1.0])
        end
    end
    return MG_CACHE[clean_path]
end

function get_island_spline(json_data, island_boundary, o_points)
    points_per_period = json_data.points_per_period
    island = json_data.island
    n_islands = json_data.n_islands
    eval_direction = json_data.eval_direction
    startϕdeg = json_data.start_phi_deg
    ix = Int(startϕdeg * 2) + 1

    risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
    ro = o_points[ix, 1]
    zo = o_points[ix, 2]
    return make_island_spline(risl, zisl, ro, zo, eval_direction)
end

function save_divertor(left_pt::Float64, right_pt::Float64, spline::ParametricSpline, json_input_files::Vector{String})
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

        # Force output divertor files to be generated directly inside target_dir
        out_name = basename(String(read_only_json.write_div_name))
        writeable_json[:write_div_name] = joinpath(target_dir, out_name)

        open(new_file_path, "w") do io
            JSON3.write(io, writeable_json)
        end

        mg = get_cached_mg(String(read_only_json.mgrid_file))
        make_classical_divertor_from_json(new_file_path; verbose=false, mg=mg)
    end
end

function run_divertor_creation(left_pt::Float64, right_pt::Float64, fwd_file_name::String, bwd_file_name::String)
    println("Creating a divertor with (theta_l, theta_r) = ($left_pt, $right_pt)")

    base_path = "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/"
    
    fwd_path = isabspath(fwd_file_name) ? fwd_file_name : joinpath(base_path, fwd_file_name)
    bwd_path = isabspath(bwd_file_name) ? bwd_file_name : joinpath(base_path, bwd_file_name)
    json_input_files = [fwd_path, bwd_path]

    # Calculate spline once using template JSON configuration
    first_json = JSON3.read(json_input_files[1])
    boundary_file = strip(String(first_json.island_boundary_file), ['"', '\'', ' '])
    opoints_file = strip(String(first_json.o_points_file), ['"', '\'', ' '])
    
    island_boundary = readdlm(boundary_file, ',')
    o_points = get_o_points(opoints_file, first_json.island, first_json.n_islands)
    spline = get_island_spline(first_json, island_boundary, o_points)

    save_divertor(left_pt, right_pt, spline, json_input_files)
end

end # module