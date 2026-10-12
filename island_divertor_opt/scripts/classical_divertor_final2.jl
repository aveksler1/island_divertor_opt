using MagneticFieldToolkit
using LinearAlgebra
using StaticArrays
using CoordinateTransformations
using Roots
using DelimitedFiles
using Dierckx
using JSON3
using Statistics
using Plots

"""    
    Given a file of o-points in `filename`, an `island_number` and the
    number of islands `n_islands` return the relevant section
    of the o-points file that corresponds to the island in question
"""
function get_o_points(filename, island_number, n_islands)
    data = readdlm(filename,',')
    npoints = div(size(data)[1], n_islands)
    start_ix = (island_number - 1) * npoints + 1
    end_ix = (island_number) * npoints
    return data[start_ix:end_ix, :]
end

"""
    Gets the appropriate section of the island boundary given the csv data
    the appropriate index, the island_number, the number of islands
    and the number of points in a period, `npoints`
"""
function get_island_boundary(all_data, ix, island_number, n_islands, npoints)
    startix = (island_number - 1) * npoints + ix
    n_skip = npoints*n_islands
    r = all_data[startix:n_skip:end,1]
    z = all_data[startix:n_skip:end,2]
    return r,z
end

"""
    This makes the spline for the island
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
    push!(θs, θs[1] + 2*pi)
    p = hcat(risl, zisl)'
    spl = ParametricSpline(θs, p, periodic=true)
    return spl
end

"""
    Given a spline, `spl` and a point (`rt`, `zt`) return
    the point on the spline that is closest.
"""
function get_closest(spl, rt, zt, ro, zo, eval_direction)
    θguess = atan(zt-zo, rt-ro)
    θguess = mod(θguess, 2π)
    function res(θ)
        r, z = spl(θ)
        return (r-rt)^2 + (z-zt)^2
    end
    lower_bound = max(0.0, θguess - π/4)
    upper_bound = min(2π, θguess + π/4)
    while upper_bound - lower_bound > 1E-6
        θrange = range(lower_bound, upper_bound, 10)
        results = res.(θrange)
        ix = argmin(results)
        lower_bound = θrange[max(1, ix-2)]
        upper_bound = θrange[min(10, ix+2)]
    end
    
    return (upper_bound + lower_bound)/2
end

function eval_point(x::Real, xyzbase::AbstractVector, x0deg::AbstractVector, xfar::AbstractVector, mg; target=3, direction=1)
    check_point = (x * xfar) + ((1-x) * x0deg)
    vec = (check_point - xyzbase)/norm(check_point - xyzbase)*direction
    halfwaypoint = 0.5*xyzbase + 0.5*check_point
    bvec = compute_magnetic_field(mg, halfwaypoint)
    bvec /= norm(bvec)
    target_sine = sin((90-target)*π/180)
    return dot(vec, bvec) - target_sine
end

function eval_point(θ::Real, ϕ::Real, xyzbase::AbstractVector, spl::ParametricSpline, mg; target=3, direction=1)
    checkr, checkz = spl(θ)
    check_point_cyl = Cylindrical(checkr, ϕ, checkz)
    check_point = CartesianFromCylindrical()(check_point_cyl)
    vec = (check_point - xyzbase)/norm(check_point - xyzbase)*direction
    halfwaypoint = 0.5*xyzbase + 0.5*check_point
    bvec = compute_magnetic_field(mg, halfwaypoint)
    bvec /= norm(bvec)
    target_sine = sin((90 - target)*π/180)
    return dot(vec, bvec) - target_sine
end

"""
    find_cone_intersections(...)

Evaluates whether the 3D space curve passing through the island scatter points
intersects the conical surface. Returns a vector of Cartesian coordinates for any
detected intersection points.
"""
function find_cone_intersections(
    X_curve::AbstractVector,
    Y_curve::AbstractVector,
    Z_curve::AbstractVector,
    xyz_center::AbstractVector,
    bhat::AbstractVector,
    theta_cone::Real,
    L::Real;
    subsamples_per_segment::Int = 15
)
    pts = [[X_curve[k], Y_curve[k], Z_curve[k]] for k in 1:length(X_curve)]
    push!(pts, pts[1]) # Close loop

    tan_th = tan(theta_cone)
    intersect_pts = Vector{Vector{Float64}}()

    # Evaluates signed distance g(p) = r_perp - h * tan(theta)
    function cone_val(p)
        d = p - xyz_center
        h = dot(d, bhat)
        r_perp = sqrt(max(0.0, norm(d)^2 - h^2))
        g = r_perp - h * tan_th
        return h, g
    end

    # Dense sampling along piecewise linear curve segments
    dense_pts = Vector{Vector{Float64}}()
    for k in 1:(length(pts)-1)
        pA, pB = pts[k], pts[k+1]
        for t in range(0, 1, length=subsamples_per_segment)
            push!(dense_pts, (1-t)*pA + t*pB)
        end
    end

    # Root finding for cone boundary zero-crossings
    for k in 1:(length(dense_pts)-1)
        p1, p2 = dense_pts[k], dense_pts[k+1]
        h1, g1 = cone_val(p1)
        h2, g2 = cone_val(p2)

        if (0.0 <= h1 <= L) || (0.0 <= h2 <= L)
            if g1 * g2 <= 0.0 && g1 != g2
                t_zero = abs(g1) / (abs(g1) + abs(g2))
                p_int = (1 - t_zero)*p1 + t_zero*p2
                h_int, _ = cone_val(p_int)
                if 0.0 <= h_int <= L
                    push!(intersect_pts, p_int)
                end
            end
        end
    end

    return intersect_pts
end

"""
    plot_diagnostic_cone(...)

Generates a 360-degree MP4 animation (in 5° increments) of the cone:
- Scatter plots island points.
- Checks and visually highlights curve intersections with the 1.5x cone.
"""
function plot_diagnostic_cone(
    i::Int,
    xyz_lc::AbstractVector,
    xyz_rc::AbstractVector,
    ϕc::Real,
    ϕn::Real,
    island_boundary,
    ix_curr::Int,
    ix_next::Int,
    island::Int,
    n_islands::Int,
    points_per_period::Int,
    mg,
    ntarget::Real,
    eval_direction::Int;
    is_error::Bool = false
)
    if !is_error
        return nothing
    end

    # Island cross section geometries
    risl_i, zisl_i = get_island_boundary(island_boundary, ix_curr, island, n_islands, points_per_period)
    X_i = risl_i .* cos(ϕc)
    Y_i = risl_i .* sin(ϕc)
    Z_i = zisl_i

    risl_i1, zisl_i1 = get_island_boundary(island_boundary, ix_next, island, n_islands, points_per_period)
    X_i1 = risl_i1 .* cos(ϕn)
    Y_i1 = risl_i1 .* sin(ϕn)
    Z_i1 = zisl_i1

    # Continuous closed loop arrays for curve checking
    X_i1_c = push!(copy(X_i1), X_i1[1])
    Y_i1_c = push!(copy(Y_i1), Y_i1[1])
    Z_i1_c = push!(copy(Z_i1), Z_i1[1])

    prefix = "diagnostic_error_step_$i"
    bases = [(xyz_lc, "Left", :orange), (xyz_rc, "Right", :magenta)]
    cone_data = Dict()

    for (xyzbase, tag, col) in bases
        cyl_base = CylindricalFromCartesian()(xyzbase)
        rc, zc = cyl_base.r, cyl_base.z

        # Trace field line to next plane phi_n
        step_ft = (ϕn - ϕc)
        a_ft = follow_field(mg, [rc, ϕc, zc], ϕn, ϕ_step = abs(step_ft)/4)
        r_ft, z_ft = a_ft.u[end][1], a_ft.u[end][2]
        xyz_ft = CartesianFromCylindrical()(Cylindrical(r_ft, ϕn, z_ft))

        # Midpoint origin calculation
        dists = [(X_i1[j] - xyz_ft[1])^2 + (Y_i1[j] - xyz_ft[2])^2 + (Z_i1[j] - xyz_ft[3])^2 for j in 1:length(X_i1)]
        k_min = argmin(dists)
        xyz_next = [X_i1[k_min], Y_i1[k_min], Z_i1[k_min]]
        xyz_center = 0.5 * (xyzbase + xyz_next)

        # B-field vector
        bvec = compute_magnetic_field(mg, xyz_center)
        bvec_norm = norm(bvec)
        bhat = bvec_norm > 0 ? bvec / bvec_norm : [0.0, 0.0, 1.0]

        cyl_n_ref = CartesianFromCylindrical()(Cylindrical(sqrt(xyzbase[1]^2+xyzbase[2]^2), ϕn, xyzbase[3]))
        if dot(bhat, cyl_n_ref - xyzbase) < 0
            bhat = -bhat
        end

        # 1.5x cone length
        R_val = sqrt(xyz_center[1]^2 + xyz_center[2]^2)
        L = 1.5 * max(0.01, R_val * abs(ϕn - ϕc))
        b_end = xyz_center + L * bhat

        # Vector basis for cone surface
        v_aux = abs(bhat[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
        uhat = cross(v_aux, bhat)
        uhat = uhat / norm(uhat)
        vhat = cross(bhat, uhat)

        theta_cone = ntarget * π / 180
        tan_th = tan(theta_cone)

        s_vals = range(0, L, length=30)
        u_vals = range(0, 2π, length=50)

        X_cone = [xyz_center[1] + s*bhat[1] + s*tan_th*(cos(u)*uhat[1] + sin(u)*vhat[1]) for s in s_vals, u in u_vals]
        Y_cone = [xyz_center[2] + s*bhat[2] + s*tan_th*(cos(u)*uhat[2] + sin(u)*vhat[2]) for s in s_vals, u in u_vals]
        Z_cone = [xyz_center[3] + s*bhat[3] + s*tan_th*(cos(u)*uhat[3] + sin(u)*vhat[3]) for s in s_vals, u in u_vals]

        # Intersection check between island i+1 curve and cone surface
        intersections = find_cone_intersections(X_i1, Y_i1, Z_i1, xyz_center, bhat, theta_cone, L)
        
        cone_data[tag] = (xyz_center, b_end, X_cone, Y_cone, Z_cone, col, xyzbase, intersections)
    end

    # Render 360-degree rotating MP4 video
    for (tag, (xyz_center, b_end, X_cone, Y_cone, Z_cone, col, xyzbase, intersections)) in cone_data
        xmin, xmax = extrema(X_cone)
        ymin, ymax = extrema(Y_cone)
        zmin, zmax = extrema(Z_cone)

        dx = max(0.005, 0.25 * (xmax - xmin))
        dy = max(0.005, 0.25 * (ymax - ymin))
        dz = max(0.005, 0.25 * (zmax - zmin))

        has_intersect = !isempty(intersections)
        status_str = has_intersect ? " [INTERSECT DETECTED]" : " [NO INTERSECTION]"
        
        if has_intersect
            println("WARNING: $tag cone intersects Island i+1 boundary curve at $(length(intersections)) locations!")
        end

        anim = @animate for az in 0:5:355
            plot(
                title = "ERROR Step $i Zoom: $tag Cone$status_str",
                xlabel = "X [m]", ylabel = "Y [m]", zlabel = "Z [m]",
                xlims = (xmin - dx, xmax + dx),
                ylims = (ymin - dy, ymax + dy),
                zlims = (zmin - dz, zmax + dz),
                camera = (az, 30),
                legend = :topright,
                size = (800, 700)
            )
            # Scatter plot island points
            scatter!(X_i, Y_i, Z_i, label="Island i Points", markersize=3, color=:blue, alpha=0.8)
            scatter!(X_i1, Y_i1, Z_i1, label="Island i+1 Points", markersize=3, color=:red, alpha=0.8)
            
            # Faint dashed line showing the curve through scatter points
            plot!(X_i1_c, Y_i1_c, Z_i1_c, label="", lw=1, linestyle=:dash, color=:red, alpha=0.4)

            # Base, Center, and B-field vector
            scatter!([xyzbase[1]], [xyzbase[2]], [xyzbase[3]], color=:grey, markersize=4, label="Base Point")
            scatter!([xyz_center[1]], [xyz_center[2]], [xyz_center[3]], color=:black, markersize=5, label="Midpoint Origin")
            plot!([xyz_center[1], b_end[1]], [xyz_center[2], b_end[2]], [xyz_center[3], b_end[3]], lw=3, color=:black, label="B-Field Vector")
            surface!(X_cone, Y_cone, Z_cone, opacity=0.35, color=col, label="$tag Cone (1.5x L)")

            # Highlight intersection points on curve
            if has_intersect
                X_int = [p[1] for p in intersections]
                Y_int = [p[2] for p in intersections]
                Z_int = [p[3] for p in intersections]
                scatter!(X_int, Y_int, Z_int, color=:yellow, markersize=7, shape=:star5, label="Cone Intersections")
            end
        end

        zoom_fname = "$(prefix)_zoom_$(lowercase(tag)).mp4"
        mp4(anim, zoom_fname, fps=15)
        println("Saved 360-degree diagnostic MP4 video: $zoom_fname")
    end

    return nothing
end

"""
Helper function to compute midpoint origin `xyz_center` from base point to island i+1 boundary
"""
function compute_midpoint_center(xyzbase::AbstractVector, ϕn::Real, spl_next::ParametricSpline, mg, step::Real)
    cyl_base = CylindricalFromCartesian()(xyzbase)
    a_ft = follow_field(mg, [cyl_base.r, cyl_base.phi, cyl_base.z], ϕn, ϕ_step = abs(step)/4)
    r_ft, z_ft = a_ft.u[end][1], a_ft.u[end][2]
    
    # Closest theta on spline i+1
    ro, zo = mean(spl_next.coefs[1, :]), mean(spl_next.coefs[2, :])
    θ_ft = get_closest(spl_next, r_ft, z_ft, ro, zo, 1)
    
    r_next, z_next = spl_next(θ_ft)
    xyz_next = CartesianFromCylindrical()(Cylindrical(r_next, ϕn, z_next))
    
    return 0.5 * (xyzbase + xyz_next)
end

"""
    make_simple_divertor(json_data, mg, island_boundary, o_points; verbose = true)
"""
function make_simple_divertor(json_data, mg, island_boundary, o_points; 
    verbose = true)
    points_per_period = json_data.points_per_period
    island = json_data.island
    n_islands = json_data.n_islands
    nfp = mg.nfp
    direction = json_data.direction
    eval_direction = json_data.eval_direction
    npoints = json_data.npoints
    startϕdeg = json_data.start_phi_deg
    mode = json_data.mode

    stepdeg = 360/nfp/points_per_period 

    r1 = json_data.r1
    z1 = json_data.z1
    r2 = json_data.r2
    z2 = json_data.z2

    target = json_data.target
    reach_target = json_data.reach_target
    if reach_target > npoints
        reach_target = npoints
    end

    rs_right = zeros(npoints+1)
    zs_right = zeros(npoints+1)
    rs_left = zeros(npoints+1)
    zs_left = zeros(npoints+1)
    phis = zeros(npoints+1)
    
    ix = Int(startϕdeg*2) + 1

    step = stepdeg*π/180*direction
    startϕ = startϕdeg * π/180

    risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
    ro = o_points[ix,1]
    zo = o_points[ix,2]
    spl = make_island_spline(risl, zisl, ro, zo, eval_direction)
    
    θl = get_closest(spl, r1, z1, ro, zo, eval_direction)
    θr = get_closest(spl, r2, z2, ro, zo, eval_direction)
    if verbose
        println("orig start point 1: ", r1," ",z1)
        println("orig start point 2: ", r2," ",z2)
    end
    
    r1 = spl(θl)[1]
    r2 = spl(θr)[1]
    z1 = spl(θl)[2]
    z2 = spl(θr)[2]
        
    if verbose
        println("real start point 1: ", r1," ",z1)
        println("real start point 2: ", r2," ",z2)
    end

    rs_right[1] = r2
    zs_right[1] = z2
    rs_left[1] = r1
    zs_left[1] = z1
    phis[1] = startϕ
    
    rc1 = r1
    zc1 = z1
    rc2 = r2
    zc2 = z2
    ϕc = startϕ
    cyl_lc = Cylindrical(rc1, ϕc, zc1)
    cyl_rc = Cylindrical(rc2, ϕc, zc2)
    
    target_arr = ones(npoints) .* target
    if reach_target > 1
        target_arr[1:reach_target] = collect(range(0.1, target, reach_target))
    end

    for i in 1:npoints
        ntarget = target_arr[i]
        
        if verbose        
            println("target = ",ntarget)
        end

        xyz_lc = CartesianFromCylindrical()(cyl_lc)
        xyz_rc = CartesianFromCylindrical()(cyl_rc)

        ϕend1 = ϕc + step
        ϕn = ϕc + step

        a1 = follow_field(mg, [rc1, ϕc, zc1], ϕend1, ϕ_step = abs(step)/4)
        r1_0deg = a1.u[end][1]
        z1_0deg = a1.u[end][2]
        cyl_ln_0deg = Cylindrical(r1_0deg, ϕn, z1_0deg)
        if verbose
            println("left point", cyl_ln_0deg)
        end

        a1 = follow_field(mg, [rc2, ϕc, zc2], ϕend1, ϕ_step = abs(step)/4)
        r2_0deg = a1.u[end][1]
        z2_0deg = a1.u[end][2]
        cyl_rn_0deg = Cylindrical(r2_0deg, ϕn, z2_0deg)
        if verbose
            println("right point", cyl_rn_0deg)
        end

        ix_curr = ix
        ix += direction
        ix_next = ix
        if verbose
            println("ix: ",ix)
        end
        
        risl, zisl = get_island_boundary(island_boundary, ix_next, island, n_islands, points_per_period)
        ro = o_points[ix_next, 1]
        zo = o_points[ix_next, 2]
        spl = make_island_spline(risl, zisl, ro, zo, eval_direction)

        θl = get_closest(spl, r1_0deg, z1_0deg, ro, zo, eval_direction)
        θr = get_closest(spl, r2_0deg, z2_0deg, ro, zo, eval_direction)

# Compute aligned midpoint origins for left and right rays
        xyz_center_l = compute_midpoint_center(xyz_lc, ϕn, spl, mg, step)
        xyz_center_r = compute_midpoint_center(xyz_rc, ϕn, spl, mg, step)

        function fleft(x)
            return eval_point(x, ϕn, xyz_center_l, spl, mg,
                              target=ntarget, direction=eval_direction)
        end
        function fright(x)
            return eval_point(x, ϕn, xyz_center_r, spl, mg,
                              target=ntarget, direction=eval_direction)
        end

        # Robust bracketed root search around initial guesses θl and θr
        θln = nothing
        θrn = nothing

        try
            # Search a ±30° window for sign changes f(θ_a) * f(θ_b) <= 0
            search_window = range(-π/6, π/6, length=60)
            
            # Left root bracket search
            vals_l = [fleft(θl + δ) for δ in search_window]
            idx_l = findfirst(k -> vals_l[k] * vals_l[k+1] <= 0, 1:length(vals_l)-1)
            if idx_l !== nothing
                θln = find_zero(fleft, (θl + search_window[idx_l], θl + search_window[idx_l+1]), Bisection())
            else
                # Fallback to local solver if no bracket found in window
                θln = find_zero(fleft, θl, Order2())
            end

            # Right root bracket search
            vals_r = [fright(θr + δ) for δ in search_window]
            idx_r = findfirst(k -> vals_r[k] * vals_r[k+1] <= 0, 1:length(vals_r)-1)
            if idx_r !== nothing
                θrn = find_zero(fright, (θr + search_window[idx_r], θr + search_window[idx_r+1]), Bisection())
            else
                θrn = find_zero(fright, θr, Order2())
            end
        catch err
            println("Error during root solve at step $i (ix=$ix_next): $err")
            plot_diagnostic_cone(
                i, xyz_lc, xyz_rc, ϕc, ϕn,
                island_boundary, ix_curr, ix_next,
                island, n_islands, points_per_period,
                mg, ntarget, eval_direction, is_error=true
            )
            rethrow(err)
        end

        cyl_lc = Cylindrical(spl(θln)[1], ϕn, spl(θln)[2])
        cyl_rc = Cylindrical(spl(θrn)[1], ϕn, spl(θrn)[2])

        rc1 = cyl_lc.r
        zc1 = cyl_lc.z
        rc2 = cyl_rc.r
        zc2 = cyl_rc.z
        
        ϕc = ϕn
        rs_right[i+1] = rc2
        rs_left[i+1] = rc1
        zs_right[i+1] = zc2
        zs_left[i+1] = zc1
        phis[i+1] = ϕc

        if verbose
            println("--------------------------")
        end
    end    
    return rs_left, zs_left, rs_right, zs_right, phis
end

function close_divertor(wing_rl, wing_rr, wing_zl, wing_zr, close)
    vec = [wing_rl - wing_rr, wing_zl - wing_zr]
    if vec[2] < 0
        vec = [wing_rr - wing_rl, wing_zr - wing_zl]
    end
    vecnorm = sqrt(sum(vec.^2))
    vec = vec / vecnorm
    vecrot = [-vec[2], vec[1]]
    rc = zeros(3)
    zc = zeros(3)
    rc[1] = wing_rr + vecrot[1]*close
    zc[1] = wing_zr + vecrot[2]*close
    rc[3] = wing_rl + vecrot[1]*close
    zc[3] = wing_zl + vecrot[2]*close
    rc[2] = (rc[1]+rc[3])/2
    zc[2] = (zc[1]+zc[3])/2
    return rc, zc
end

function get_divertor_coords(rr, rl, zr, zl, left_extension, right_extension)
    vec = [rr - rl, zr - zl]
    veclen = norm(vec)
    vecnorm = vec/veclen
    rdiv = zeros(5)
    zdiv = zeros(5)
    p_array = [-left_extension, 0.0, 0.5, 1.0, 1 + right_extension]
    for k in 1:5
        p = p_array[k]
        if p < 0 
            mult = p * vecnorm
            rdiv[k], zdiv[k] = [rl, zl] .+ mult
        elseif p > 1 
            mult = (p-1) * vecnorm
            rdiv[k], zdiv[k] = [rr, zr] .+ mult
        elseif p == 0.5
            rdiv[k] = (rr+rl)/2
            zdiv[k] = (zr+zl)/2
        elseif p < 0.5
            mult = p * vecnorm
            rdiv[k], zdiv[k] = [rl, zl] .+ mult
        elseif p > 0.5
            mult = (p-1) *vecnorm
            rdiv[k], zdiv[k] = [rr, zr] .+ mult
        end
    end
    return rdiv, zdiv
end

function write_classical_divertor(wf, rl, rr, zl, zr, phi;
                         left_extension=0.05, right_extension=0.05, closed=0.0)

    if phi[end] < phi[1]
        istart = length(phi)
        iend = 1
        istep = -1
    else
        istart = 1
        iend = length(phi)
        istep = 1
    end
    for i in istart:istep:iend
        rdiv, zdiv = get_divertor_coords(rr[i], rl[i], zr[i], zl[i], left_extension, right_extension)
        write(wf, "    ")
        write(wf, string(phi[i]*180/π))
        write(wf, "\n")
        for k in 1:5
            write(wf, string(rdiv[k]*100))
            write(wf, "  ")
            write(wf, string(zdiv[k]*100))
            write(wf, "\n")
        end
        if closed > 0
            rback, zback = close_divertor(rdiv[1], rdiv[end], zdiv[1], zdiv[end], closed)
            for k in 1:3
                write(wf, string(rback[k]*100))
                write(wf, "  ")
                write(wf, string(zback[k]*100))
                write(wf, "\n")
            end
            write(wf, string(rdiv[1]*100))
            write(wf, "  ")
            write(wf, string(zdiv[1]*100))
            write(wf, "\n")
        end
    end
end

function make_classical_divertor_from_json(json_input_file; verbose=true)
    json_data = JSON3.read(json_input_file)

    if json_data.mgrid_file[end-2:end] == ".h5"
        mg = read_hint_field(json_data.mgrid_file)
    else
        mg = read_mgrid(json_data.mgrid_file, [1.0])
    end

    if hasproperty(json_data, :closed)
        closed = json_data.closed
    else
        closed = 0.0
    end

    if hasproperty(json_data, :right_extension)
        right_extension = json_data.right_extension
    else
        right_extension = 0.05
    end
    if hasproperty(json_data, :left_extension)
        left_extension = json_data.left_extension
    else
        left_extension = 0.05
    end

    island_boundary = readdlm(json_data.island_boundary_file,',')
    o_points = get_o_points(
        json_data.o_points_file, json_data.island, json_data.n_islands
    )

    rl, zl, rr, zr, phi = make_simple_divertor(
        json_data, mg, island_boundary, o_points, verbose=verbose
    )

    ntor = length(phi) 

    wf = open(json_data.write_div_name,"w")
    write(wf, "#"*json_data.write_div_tag*"\n")
    if closed > 0.0
        npol = "9"
    else
        npol = "5"
    end
    write(wf, string(ntor)*" "*npol*" "*string(mg.nfp)*" 0.0 0.0\n")

    write_classical_divertor(wf, rl, rr, zl, zr,  phi,
            right_extension=right_extension,
            left_extension=left_extension, 
            closed=closed)
    close(wf)

    if closed > 0.0
        if verbose
            println("Closing divertor...")
        end
        wall_dict, deg_arr = read_text(json_data.write_div_name)
        if json_data.direction > 0
            close_and_write_cap(
                wall_dict[deg_arr[end]][:r], deg_arr[end], wall_dict[deg_arr[end]][:z], 
                0.001, json_data.write_div_tag*"_cap"
            )
        else
            close_and_write_cap(
                wall_dict[deg_arr[begin]][:r], deg_arr[begin], wall_dict[deg_arr[begin]][:z], 
                -0.001, json_data.write_div_tag*"_cap"
            )
        end
    end
end

function read_text(file)
    local text::String
    text = read(file, String)
    text = strip(text)
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
        wall_data = Dict(:r => r, :z => z)
        wall_dict[deg_arr[i]] = wall_data
    end
    return wall_dict, deg_arr
end

function close_and_write_cap(r, phi, z, istep, str)

    function get_rz(c)
        _r = sqrt(c[1]^2 + c[2]^2)
        _z = c[3]
        return _r, _z
    end

    phi = phi * pi / 180
    rmid = mean(r)
    zmid = mean(z)

    midpt = CartesianFromCylindrical()(Cylindrical(rmid, phi, zmid))
    rcap = zeros(9)
    zcap = zeros(9)

    for i in 1:9
        c = Cylindrical(r[i], phi, z[i])
        c_cart = CartesianFromCylindrical()(c)
        vec = midpt - c_cart
        c_new = c_cart + vec*0.99 
        r_new, z_new = get_rz(c_new)
        rcap[i] = r_new
        zcap[i] = z_new
    end
    phicap = phi*(1 + istep)

    wf = open(str*".txt","w")
    write(wf, "#"*str*"\n")
    ntor = 2
    npol = 9
    write(wf, string(ntor)*" "*string(npol)*" "*string(4)*" 0.0 0.0\n")
    
    function write_points(r, phi, z)
        write(wf, "    ")
        write(wf, string(phi * 180 / pi))
        write(wf, "\n")
        for i in 1:length(r)
            write(wf, (string(r[i])))
            write(wf, "  ")
            write(wf, string(z[i]))
            write(wf, "\n")
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