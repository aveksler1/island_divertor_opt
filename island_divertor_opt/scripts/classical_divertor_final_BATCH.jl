using MagneticFieldToolkit
using LinearAlgebra
using StaticArrays
using CoordinateTransformations
using Roots
using DelimitedFiles
using Dierckx
using JSON3
using Statistics

"""    
    Given a file of o-points in `filename`, an `island_number` and the
    number of islands `n_islands` return the relevant section
    of the o-points file that corresponds to the island in question
"""
function get_o_points(filename, island_number, n_islands)
    clean_file = strip(String(filename), ['"', '\'', ' '])
    data = readdlm(clean_file, ',')
    npoints = div(size(data)[1], n_islands)
    start_ix = (island_number - 1) * npoints + 1
    end_ix = (island_number) * npoints
    return data[start_ix:end_ix, :]
end

"""
    Gets the appropriate section of the island boundary given the csv data
"""
function get_island_boundary(all_data, ix, island_number, n_islands, npoints)
    startix = (island_number - 1) * npoints + ix
    n_skip = npoints * n_islands
    r = all_data[startix:n_skip:end, 1]
    z = all_data[startix:n_skip:end, 2]
    return r, z
end

"""
    Makes a periodic spline for the island boundary
"""
function make_island_spline(risl, zisl, ro, zo, eval_direction)
    θs = atan.(zisl .- zo, risl .- ro)
    θs = mod.(θs, 2π)
    sort_indices = sortperm(θs)
    θs = θs[sort_indices]
    risl = risl[sort_indices]
    zisl = zisl[sort_indices]
    
    push!(risl, risl[1])
    push!(zisl, zisl[1])
    push!(θs, θs[1] + 2π)
    p = hcat(risl, zisl)'
    spl = ParametricSpline(θs, p, periodic=true)
    return spl
end

"""
    Finds the closest point on the spline to a target point
"""
function get_closest(spl, rt, zt, ro, zo, eval_direction)
    θguess = atan(zt - zo, rt - ro)
    θguess = mod(θguess, 2π)
    
    function res(θ)
        r, z = spl(θ)
        return (r - rt)^2 + (z - zt)^2
    end

    lower_bound = max(0.0, θguess - π/4)
    upper_bound = min(2π, θguess + π/4)
    
    while upper_bound - lower_bound > 1e-6
        θrange = range(lower_bound, upper_bound, length=10)
        results = res.(θrange)
        ix = argmin(results)
        lower_bound = θrange[max(1, ix - 2)]
        upper_bound = θrange[min(10, ix + 2)]
    end
    
    return (upper_bound + lower_bound) / 2
end

"""
    Evaluates whether a point intersects the field with the target angle
"""
function eval_point(x::Real, xyzbase::AbstractVector, x0deg::AbstractVector, xfar::AbstractVector, mg; target=3, direction=1)
    check_point = (x * xfar) + ((1 - x) * x0deg)
    vec = (check_point - xyzbase) / norm(check_point - xyzbase) * direction
    halfwaypoint = 0.5 * xyzbase + 0.5 * check_point
    bvec = compute_magnetic_field(mg, halfwaypoint)
    bvec /= norm(bvec)
    target_sine = sin((90 - target) * π / 180)
    return dot(vec, bvec) - target_sine
end

function eval_point(θ::Real, ϕ::Real, xyzbase::AbstractVector, spl::ParametricSpline, mg; target=3, direction=1)
    checkr, checkz = spl(θ)
    check_point_cyl = Cylindrical(checkr, ϕ, checkz)
    check_point = CartesianFromCylindrical()(check_point_cyl)
    vec = (check_point - xyzbase) / norm(check_point - xyzbase) * direction
    halfwaypoint = 0.5 * xyzbase + 0.5 * check_point
    bvec = compute_magnetic_field(mg, halfwaypoint)
    bvec /= norm(bvec)
    target_sine = sin((90 - target) * π / 180)
    return dot(vec, bvec) - target_sine
end

"""
    Traces field lines and generates the divertor geometry
"""
function make_simple_divertor(json_data, mg, island_boundary, o_points; verbose=true)
    points_per_period = json_data.points_per_period
    island = json_data.island
    n_islands = json_data.n_islands
    nfp = mg.nfp
    direction = json_data.direction
    eval_direction = json_data.eval_direction
    npoints = json_data.npoints
    startϕdeg = json_data.start_phi_deg
    mode = json_data.mode

    stepdeg = 360 / nfp / points_per_period 

    r1 = json_data.r1
    z1 = json_data.z1
    r2 = json_data.r2
    z2 = json_data.z2

    target = json_data.target
    reach_target = json_data.reach_target
    if reach_target > npoints
        reach_target = npoints
    end

    rs_right = zeros(npoints + 1)
    zs_right = zeros(npoints + 1)
    rs_left = zeros(npoints + 1)
    zs_left = zeros(npoints + 1)
    phis = zeros(npoints + 1)
    
    ix = Int(startϕdeg * 2) + 1
    step = stepdeg * π / 180 * direction
    startϕ = startϕdeg * π / 180

    risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
    ro = o_points[ix, 1]
    zo = o_points[ix, 2]
    spl = make_island_spline(risl, zisl, ro, zo, eval_direction)
    
    θl = get_closest(spl, r1, z1, ro, zo, eval_direction)
    θr = get_closest(spl, r2, z2, ro, zo, eval_direction)
    
    r1 = spl(θl)[1]
    r2 = spl(θr)[1]
    z1 = spl(θl)[2]
    z2 = spl(θr)[2]

    rs_right[1] = r2
    zs_right[1] = z2
    rs_left[1] = r1
    zs_left[1] = z1
    phis[1] = startϕ
    
    rc1, zc1 = r1, z1
    rc2, zc2 = r2, z2
    ϕc = startϕ
    
    cyl_lc = Cylindrical(rc1, ϕc, zc1)
    cyl_rc = Cylindrical(rc2, ϕc, zc2)
    
    target_arr = ones(npoints) .* target
    if reach_target > 1
        target_arr[1:reach_target] = collect(range(0.1, target, length=reach_target))
    end

    for i in 1:npoints
        ntarget = target_arr[i]
        
        xyz_lc = CartesianFromCylindrical()(cyl_lc)
        xyz_rc = CartesianFromCylindrical()(cyl_rc)

        ϕend1 = ϕc + step
        ϕn = ϕc + step

        a1 = follow_field(mg, [rc1, ϕc, zc1], ϕend1, ϕ_step=abs(step) / 4)
        r1_0deg = a1.u[end][1]
        z1_0deg = a1.u[end][2]

        a1 = follow_field(mg, [rc2, ϕc, zc2], ϕend1, ϕ_step=abs(step) / 4)
        r2_0deg = a1.u[end][1]
        z2_0deg = a1.u[end][2]

        ix += direction
        
        risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
        ro = o_points[ix, 1]
        zo = o_points[ix, 2]
        spl = make_island_spline(risl, zisl, ro, zo, eval_direction)

        θl = get_closest(spl, r1_0deg, z1_0deg, ro, zo, eval_direction)
        θr = get_closest(spl, r2_0deg, z2_0deg, ro, zo, eval_direction)

        fleft(x) = eval_point(x, ϕn, xyz_lc, spl, mg, target=ntarget, direction=eval_direction)
        fright(x) = eval_point(x, ϕn, xyz_rc, spl, mg, target=ntarget, direction=eval_direction)

        angle = π / 30

        if mode == 0 
            θln = find_zero(fleft, θl - angle, Order2())
            θrn = find_zero(fright, θr + angle, Order2())
        elseif mode == 1
            θln = find_zero(fleft, θl - angle, Order2())
            θrn = find_zero(fright, θr - angle, Order2())
        elseif mode == 2
            θln = find_zero(fleft, θl + angle, Order2())
            θrn = find_zero(fright, θr - angle, Order2())
        elseif mode == 3
            θln = find_zero(fleft, θl + angle, Order2())
            θrn = find_zero(fright, θr + angle, Order2())
        else
            throw(ErrorException("Invalid mode specified: $mode."))
        end

        cyl_lc = Cylindrical(spl(θln)[1], ϕn, spl(θln)[2])
        cyl_rc = Cylindrical(spl(θrn)[1], ϕn, spl(θrn)[2])

        rc1, zc1 = cyl_lc.r, cyl_lc.z
        rc2, zc2 = cyl_rc.r, cyl_rc.z
       
        ϕc = ϕn
        rs_right[i + 1] = rc2
        rs_left[i + 1] = rc1
        zs_right[i + 1] = zc2
        zs_left[i + 1] = zc1
        phis[i + 1] = ϕc
    end    
    return rs_left, zs_left, rs_right, zs_right, phis
end

function close_divertor(wing_rl, wing_rr, wing_zl, wing_zr, close)
    vec = [wing_rl - wing_rr, wing_zl - wing_zr]
    if vec[2] < 0
        vec = [wing_rr - wing_rl, wing_zr - wing_zl]
    end
    vecnorm = sqrt(sum(vec .^ 2))
    vec = vec / vecnorm
    vecrot = [-vec[2], vec[1]]
    
    rc = zeros(3)
    zc = zeros(3)
    rc[1] = wing_rr + vecrot[1] * close
    zc[1] = wing_zr + vecrot[2] * close
    rc[3] = wing_rl + vecrot[1] * close
    zc[3] = wing_zl + vecrot[2] * close
    rc[2] = (rc[1] + rc[3]) / 2
    zc[2] = (zc[1] + zc[3]) / 2
    return rc, zc
end

function get_divertor_coords(rr, rl, zr, zl, left_extension, right_extension)
    vec = [rr - rl, zr - zl]
    veclen = norm(vec)
    vecnorm = vec / veclen
    rdiv = zeros(5)
    zdiv = zeros(5)
    p_array = [-left_extension, 0.0, 0.5, 1.0, 1 + right_extension]
    for k in 1:5
        p = p_array[k]
        if p < 0 
            mult = p * vecnorm
            rdiv[k], zdiv[k] = [rl, zl] .+ mult
        elseif p > 1 
            mult = (p - 1) * vecnorm
            rdiv[k], zdiv[k] = [rr, zr] .+ mult
        elseif p == 0.5
            rdiv[k] = (rr + rl) / 2
            zdiv[k] = (zr + zl) / 2
        elseif p < 0.5
            mult = p * vecnorm
            rdiv[k], zdiv[k] = [rl, zl] .+ mult
        elseif p > 0.5
            mult = (p - 1) * vecnorm
            rdiv[k], zdiv[k] = [rr, zr] .+ mult
        end
    end
    return rdiv, zdiv
end

function write_classical_divertor(wf, rl, rr, zl, zr, phi; left_extension=0.05, right_extension=0.05, closed=0.0)
    if phi[end] < phi[1]
        istart, iend, istep = length(phi), 1, -1
    else
        istart, iend, istep = 1, length(phi), 1
    end

    for i in istart:istep:iend
        rdiv, zdiv = get_divertor_coords(rr[i], rl[i], zr[i], zl[i], left_extension, right_extension)
        write(wf, "    $(phi[i] * 180 / π)\n")
        for k in 1:5
            write(wf, "$(rdiv[k] * 100)  $(zdiv[k] * 100)\n")
        end
        if closed > 0
            rback, zback = close_divertor(rdiv[1], rdiv[end], zdiv[1], zdiv[end], closed)
            for k in 1:3
                write(wf, "$(rback[k] * 100)  $(zback[k] * 100)\n")
            end
            write(wf, "$(rdiv[1] * 100)  $(zdiv[1] * 100)\n")
        end
    end
end

"""
    Entry point to build a divertor from JSON. Accepts optional pre-loaded `mg` magnetic field.
"""
function make_classical_divertor_from_json(json_input_file; verbose=true, mg=nothing)
    json_data = JSON3.read(json_input_file)

    if mg === nothing
        mgrid_file_clean = strip(String(json_data.mgrid_file), ['"', '\'', ' '])
        if endswith(mgrid_file_clean, ".h5")
            mg = read_hint_field(mgrid_file_clean)
        else
            mg = read_mgrid(mgrid_file_clean, [1.0])
        end
    end

    closed = hasproperty(json_data, :closed) ? json_data.closed : 0.0
    right_extension = hasproperty(json_data, :right_extension) ? json_data.right_extension : 0.05
    left_extension = hasproperty(json_data, :left_extension) ? json_data.left_extension : 0.05

    boundary_file = strip(String(json_data.island_boundary_file), ['"', '\'', ' '])
    island_boundary = readdlm(boundary_file, ',')
    
    opoints_file = strip(String(json_data.o_points_file), ['"', '\'', ' '])
    o_points = get_o_points(opoints_file, json_data.island, json_data.n_islands)

    rl, zl, rr, zr, phi = make_simple_divertor(json_data, mg, island_boundary, o_points, verbose=verbose)

    ntor = length(phi) 
    out_file = strip(String(json_data.write_div_name), ['"', '\'', ' '])

    wf = open(out_file, "w")
    write(wf, "#" * json_data.write_div_tag * "\n")
    npol = closed > 0.0 ? "9" : "5"
    write(wf, "$ntor $npol $(mg.nfp) 0.0 0.0\n")

    write_classical_divertor(wf, rl, rr, zl, zr, phi, right_extension=right_extension, left_extension=left_extension, closed=closed)
    close(wf)

    if closed > 0.0
        wall_dict, deg_arr = read_text(out_file)
        out_dir = dirname(out_file)
        if isempty(out_dir)
            out_dir = pwd()
        end
        cap_prefix = joinpath(out_dir, String(json_data.write_div_tag) * "_cap")

        if json_data.direction > 0
            close_and_write_cap(
                wall_dict[deg_arr[end]][:r], deg_arr[end], wall_dict[deg_arr[end]][:z], 
                0.001, cap_prefix
            )
        else
            close_and_write_cap(
                wall_dict[deg_arr[begin]][:r], deg_arr[begin], wall_dict[deg_arr[begin]][:z], 
                -0.001, cap_prefix
            )
        end
    end
end

function read_text(file)
    text = strip(read(file, String))
    entries = split(text, '\n')
    data_first_line = split(entries[2], ' ')
    npoloidal = parse(Int, data_first_line[2])
    entries = entries[3:end]
    deg_arr = round.(parse.(Float32, entries[1:npoloidal+1:end]), digits=2)
    idx = 1
    wall_dict = Dict()
    for i in 1:length(deg_arr)
        idx += 1
        data = entries[idx:idx+(npoloidal-1)]
        idx += npoloidal
        r = map(line -> parse(Float64, split(line)[1]), data)
        z = map(line -> parse(Float64, split(line)[2]), data)
        wall_dict[deg_arr[i]] = Dict(:r => r, :z => z)
    end
    return wall_dict, deg_arr
end

function close_and_write_cap(r, phi, z, istep, cap_filepath_prefix)
    get_rz(c) = sqrt(c[1]^2 + c[2]^2), c[3]

    phi = phi * π / 180
    midpt = CartesianFromCylindrical()(Cylindrical(mean(r), phi, mean(z)))
    rcap, zcap = zeros(9), zeros(9)

    for i in 1:9
        c_cart = CartesianFromCylindrical()(Cylindrical(r[i], phi, z[i]))
        c_new = c_cart + (midpt - c_cart) * 0.99 
        rcap[i], zcap[i] = get_rz(c_new)
    end
    phicap = phi * (1 + istep)

    cap_file = cap_filepath_prefix * ".txt"
    wf = open(cap_file, "w")
    write(wf, "#" * basename(cap_filepath_prefix) * "\n")
    write(wf, "2 9 4 0.0 0.0\n")
    
    function write_points(r, phi, z)
        write(wf, "    $(phi * 180 / π)\n")
        for i in 1:length(r)
            write(wf, "$(r[i])  $(z[i])\n")
        end
    end

    if istep > 0
        write_points(r, phi, z)
        write_points(rcap, phicap, zcap)
    else
        write_points(rcap, phicap, zcap)
        write_points(r, phi, z)
    end

    close(wf)
end