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
    Given a file of o-points in `filename`, an `island_number and the
    number of islands `n_islands` return the relevant section
    of the o-points file that corresponds to the island in question
"""
function get_o_points(filename, island_number, n_islands)
    data = readdlm(filename,',')
    # The number of points per island is the number of rows divided by the
    # number of islands
    npoints = div(size(data)[1], n_islands)
    # First island would be found at first index, hence (island_number -1)
    start_ix = (island_number - 1) * npoints + 1
    # start_ix + npoints (no +1, bc that would be first index of next island)
    end_ix = (island_number) * npoints
    return data[start_ix:end_ix, :]
end

"""
    Gets the appropriate section of the island boundary given the csv data
    the appropriate index (for example if the file is sampled 1 per degree, the
    index would be the degree value), the island_number, the number of islands
    and the number of points in a period, `npoint`s
"""
function get_island_boundary(all_data, ix, island_number, n_islands, npoints)
    startix = (island_number - 1) * npoints + ix
    n_skip = npoints*n_islands
    r = all_data[startix:n_skip:end,1]
    z = all_data[startix:n_skip:end,2]
    return r,z
end

"""
    This makes the spline for the island in kind of a hacky way
    namely it uses the o-point value and it also starts the 
    θ=0 value to be on the y axis in the direction of the LCFS

    This will not work if the surfaces are very crescent shaped and
    we'll need a better splining algorithm
"""
function make_island_spline(risl, zisl, ro, zo, eval_direction)

    # Append the first element to the end of the list for risl and zisl
    # This occurs in place, maybe would be better to allocate new memory
    θs = atan.(zisl .- zo, risl .- ro) # element wise subtraction
    # offset 90 degrees so that seam isn't in a place we care about
    # θs .+= (eval_direction*π/2)
    θs = mod.(θs, 2π)
    sort_indices = sortperm(θs)
    θs = θs[sort_indices]
    risl = risl[sort_indices]
    zisl = zisl[sort_indices]
    # Adjusting arrays to make spline periodic
    push!(risl, risl[1])
    push!(zisl, zisl[1])
    push!(θs, θs[1] + 2*pi)
    p = hcat(risl, zisl)' # Dash = transpose
    spl = ParametricSpline(θs, p, periodic=true)
    return spl
end

"""
    Given a spline, `spl` and a point (`rt`, `zt`) return
    the point on the spline that is closest.

    It initializes a guess for theta by using the `ro`, `zo` point
    and takes into account that the spline starts θ on the y axis
"""
function get_closest(spl, rt, zt, ro, zo, eval_direction)
    #get initial guess for theta
    #println(rt,", ",zt,", ",ro,", ",zo)
    θguess = atan(zt-zo, rt-ro)
    # θguess += eval_direction*π/2 # this line will get removed if i take out the seam shifting part
    θguess = mod(θguess, 2π)
    function res(θ)
        r, z = spl(θ)
        return (r-rt)^2 + (z-zt)^2
    end
    #write my own goddamn optimizer bc i wasted 90 minutes trying to figure out
    #how to do this simple fucking problem in julia with no success
    #println("theta_guess: ",θguess," ",res(θguess))
    #get lower bound and upper bound
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

"""
    eval_point(x, xyzbase, x0deg, xfar, mg, target=3, direction=1)
    eval_point(θ, ϕ, xyzbase, spl, mg, target=3, direction=1)

    This code checks whether a point intersects the field with the target angle and is used for optimization
    There are two forms, one that evaluates a point on the line between two given points in the next plane
    and one that evaluates a point on an interpolated spline

    The geometry is as follows
    `x` is the thing you're evaluating.  It's a number between 0 and 1. Draw a line between
        `x0deg` and `xfar`.  and then move along that line such that 0 would be at `x0deg` and \
        1 would be at `xfar`. That is the point you are testing
    `xyzbase` is the point on the divertor plane at the previous phi value (say 0 degrees)
    `x0deg` is the point at the current phi plane (say 1 degree) that is made by following that field line to
        the target plane. It's called 0deg because this will make an angle of close to 0 degrees
        with the field if you were to put the divertor here
    `xfar` is the point on the current phi plane (say 1 degree) that makes an angle larger than the target angle


    The evaluation returns the distance between the angle the field line will make with a line that goes from
    `xyzbase` to the point given by `x` to the target angle (given in degrees)

    in the spline version:
    `θ` is the value independent parameter of the spline that you are evaluating
    `ϕ` is the toroidal angle to evaluate on
    `xyzbase` is the xyz coordinate of the point in the previous phi plane
    `spl` is the spline at ϕ

"""
function eval_point(x::Real, xyzbase::AbstractVector, x0deg::AbstractVector, xfar::AbstractVector, mg; target=3, direction=1)
    #calculate a point that is x distance from 0deg and far
    # This is the same as x0 + x(xfar - x0), which is like moving in the direction 
    # from x0 to xfar (vector) by a distance x starting from x0. 
    check_point = (x * xfar) + ((1-x) * x0deg)
    # Calculates a vector from previous divertor point to checkpoint
    vec = (check_point - xyzbase)/norm(check_point - xyzbase)*direction
    # Calculates the half way pt and computes the magnetic field at the halfway
    # point.
    halfwaypoint = 0.5*xyzbase + 0.5*check_point
    bvec = compute_magnetic_field(mg, halfwaypoint)
    bvec /= norm(bvec)
    # Compares the angle made from vec to checkpoint to the target angle. 
    # If they line up, return 0. 
    target_sine = sin((90-target)*π/180)
    return dot(vec, bvec) - target_sine
end


function eval_point(θ::Real, ϕ::Real, xyzbase::AbstractVector, spl::ParametricSpline, mg; target=3, direction=1)
    # calculate a point that is x distance from 0deg and far
    # Get r and z points from the theta angle given the next island spline
    checkr, checkz = spl(θ)
    # Create a cylindrical point
    check_point_cyl = Cylindrical(checkr, ϕ, checkz)
    # Convert the cylindrical point to cartesian
    check_point = CartesianFromCylindrical()(check_point_cyl)
    # Create a vector from previous divertor point to checkpoint, and 
    # normalize it. 
    vec = (check_point - xyzbase)/norm(check_point - xyzbase)*direction
    # Compute the halfway point and the magnetic field at the halfway point
    halfwaypoint = 0.5*xyzbase + 0.5*check_point
    bvec = compute_magnetic_field(mg, halfwaypoint)
    bvec /= norm(bvec)
    # We want that angle between the two vectors to be the target. The dot
    # products yields cos(angle). So, we want eval_point to return 0 when 
    # the dot product between vec and bvec is equivalent to cos(target), or 
    # in this case coded as sin(90 - target)
    target_sine = sin((90 - target)*π/180)
    return dot(vec, bvec) - target_sine
end

"""
    make_simple_divertor(mg, direction; target=3, npoints=22, reach_target=1)

Inputs 
    `mg`: magnetic field grid, could be from BMW, HINT, mgrid, julia don't care
optional inputs
    `stepdeg`: distance between divertor points (in degrees, default 0.5)
    `verbose`: whether to print out diagnostic info, default true

Outputs:
    `rs_left`: left (r) points on the divertor
    `zs_left`: left (z) points on the divertor
    `rs_right`: right (r) points on the divertor
    `zs_right`: right (z) points on the divertor
    `rs_center`: center (r) points on the divertor
    `zs_center`: center (z) points on the divertor
    `phis`: the phi values corresponding to the planes

Note, due to an early convention "left" is the first point on the divertor plate 
when moving in the counter-clockwise direction. It would indeed be left if the divertor is on the bottom
In the json input settings, "left" corresponds to "1"

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


    #calculate step degree from information
    stepdeg = 360/nfp/points_per_period 

    # starting values of divertor
    r1 = json_data.r1
    z1 = json_data.z1
    r2 = json_data.r2
    z2 = json_data.z2

    # Ensures that round off will maximum be the entire length of the divertor
    target = json_data.target
    reach_target = json_data.reach_target
    if reach_target > npoints
        reach_target = npoints
    end

    # These are the vectors that will hold the R and Z values for the right and 
    # left of the divertor
    rs_right = zeros(npoints+1)
    zs_right = zeros(npoints+1)
    rs_left = zeros(npoints+1)
    zs_left = zeros(npoints+1)
    phis = zeros(npoints+1)
    
    # The index for getting values, this should be made more general for 
    # non-half degrees
    ix = Int(startϕdeg*2) + 1

    step = stepdeg*π/180*direction # step size in radians with direction (fwd or bwd)
    startϕ = startϕdeg * π/180 # starting phi in radians

    # Getting the spline of the island boundary
    risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
    # (r, z) of O-point
    ro = o_points[ix,1]
    zo = o_points[ix,2]
    # makes a spline of the island boundary
    spl = make_island_spline(risl, zisl, ro, zo, eval_direction)
    
    # Gets the value of (r1, z1) and (r2, z2) on the island boundary
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
    # Computing vector along width of divertor
    # along_div_vec = [r1-r2, z1-z2]
    # along_div_vec = along_div_vec / norm(along_div_vec)
        
    if verbose
        println("real start point 1: ", r1," ",z1)
        println("real start point 2: ", r2," ",z2)
    end

    #initialize some values before the loops
    # Starting values of the divertor array are the starting points we pass in
    rs_right[1] = r2
    zs_right[1] = z2
    rs_left[1] = r1
    zs_left[1] = z1
    phis[1] = startϕ
    
    # Setting loop vars as starting vars initialy 
    rc1 = r1
    zc1 = z1
    rc2 = r2
    zc2 = z2
    ϕc = startϕ
    # Geting cylindrical coords for starting points
    cyl_lc = Cylindrical(rc1, ϕc, zc1)
    cyl_rc = Cylindrical(rc2, ϕc, zc2)
    
    # This handles the ramping of the angle before hitting the target for field intersection
    # It starts at 0, and linearly increases to the target angle based on the number of steps
    target_arr = ones(npoints) .* target
    # This means that the first reach_target points will ramp up to the target
    # value. This is what Aaron means by the divertor rounding off at the top.

    # Does this get handled correctly for the backwards divertor? It seems like 
    # it might be making the target angle 
    if reach_target > 1
        target_arr[1:reach_target] = collect(range(0.1, target, reach_target))
    end

    # The main loop.  Solve for the divertor at each phi step, given the 
    # position in the previous step
    for i in 1:npoints
        ntarget = target_arr[i]
        
        if verbose        
            println("target = ",ntarget)
        end

        # Initial points in xyz for the current phi plane
        xyz_lc = CartesianFromCylindrical()(cyl_lc)
        xyz_rc = CartesianFromCylindrical()(cyl_rc)

        #Step 1
        #We calculate three points in the next phi plane.  These are the points if you just moved
        #the divertor point to the phi plane along the field. If the divertor was placed here the
        #intercepting angle would be about 0 degrees (which we don't want)
        
        # Step forward in phi
        ϕend1 = ϕc + step
        ϕn = ϕc + step

        # Compute left point on island boundary where the field line goes
        a1 = follow_field(mg, [rc1, ϕc, zc1], ϕend1, ϕ_step = abs(step)/4)
        r1_0deg = a1.u[end][1]
        z1_0deg = a1.u[end][2]
        cyl_ln_0deg = Cylindrical(r1_0deg, ϕn, z1_0deg)
        if verbose
            println("left point", cyl_ln_0deg)
        end

        # Do the same thing but for the right points
        a1 = follow_field(mg, [rc2, ϕc, zc2], ϕend1, ϕ_step = abs(step)/4)
        r2_0deg = a1.u[end][1]
        z2_0deg = a1.u[end][2]
        cyl_rn_0deg = Cylindrical(r2_0deg, ϕn, z2_0deg)
        if verbose
            println("right point", cyl_rn_0deg)
        end

        # generate the splines for the next phi plane
        ix += direction
        if verbose
            println("ix: ",ix)
        end
        
        risl, zisl = get_island_boundary(island_boundary, ix, island, n_islands, points_per_period)
        ro = o_points[ix, 1]
        zo = o_points[ix, 2]
        spl = make_island_spline(risl, zisl, ro, zo, eval_direction)

        # Get the θ values that are closest to the left and right points where
        # the field line ends up on the island boundary at the next phi plane
        θl = get_closest(spl, r1_0deg, z1_0deg, ro, zo, eval_direction)
        θr = get_closest(spl, r2_0deg, z2_0deg, ro, zo, eval_direction)

        #helper functions for the searching
        function fleft(x)
            return eval_point(x, ϕn, xyz_lc, spl, mg,
                              target=ntarget, direction=eval_direction)
        end
        function fright(x)
            return eval_point(x, ϕn, xyz_rc, spl, mg,
                              target=ntarget, direction=eval_direction)
        end


        # Use optimization algorithms to find the left and right points of the 
        # divertor.
        # Finds at which point that fleft/fright return 0, aka when the angle
        # between the divertor at the previous plane and the current plane is 
        # equal to the target value. 
        angle = π/60

        θln = nothing
        θrn = nothing

        if mode == 0 
            # Left point -, right point +
            θln = find_zero(fleft, θl-angle, Order2())
            θrn = find_zero(fright, θr+angle, Order2())
        elseif mode == 1
            # # Left point -, right point - 
            θln = find_zero(fleft, θl-angle, Order2())
            θrn = find_zero(fright, θr-angle, Order2())
        elseif mode == 2
            # Left point +, right point - 
            θln = find_zero(fleft, θl+angle, Order2())
            θrn = find_zero(fright, θr-angle, Order2())
        elseif mode == 3
            # Left point +, right point +
            θln = find_zero(fleft, θl+angle, Order2())
            θrn = find_zero(fright, θr+angle, Order2())
        else
            throw(ErrorException("Invalid mode specified: $mode. Mode must be between 0 and 3."))
        end
        # ---- This introduces an unwanted self-shadowing effect that we should
        # check how bad it is, but is probably unwanted. Hopefully the fix
        # in the periodicity of the spline resolves this issue.
        # try 
        #     θln = find_zero(fleft, θl-angle, Order2())
        # catch 
        #     θln = find_zero(fleft, θl+angle, Order2())
        # finally
        #     println("Failed to converge on $ix for left point")
        # end

        # try 
        #     θrn = find_zero(fright, θr-angle, Order2())
        # catch 
        #     θrn = find_zero(fright, θr+angle, Order2())
        # finally
        #     println("Failed to converge on $ix for right point")
        # end


        cyl_lc = Cylindrical(spl(θln)[1], ϕn, spl(θln)[2])
        cyl_rc = Cylindrical(spl(θrn)[1], ϕn, spl(θrn)[2])

        # Set things up for the next step of the iteration. This sets up the 
        # next starting points to be the just calculated points. 
        rc1 = cyl_lc.r
        zc1 = cyl_lc.z
        rc2 = cyl_rc.r
        zc2 = cyl_rc.z
       
        # Store these values in the output array. 
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

"""
    function to step back some distance from the divertor and make a closed divertor structure.

    Need to add closing on the front and back plates too by adding smaller rectangle and building
    tris. 
"""    
function close_divertor(wing_rl, wing_rr, wing_zl, wing_zr, close)
    vec = [wing_rl - wing_rr, wing_zl - wing_zr]
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

"""
Function to get the divertor coords 
"""
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

"""
    function to write the divertor output in Kisslinger format

    Basic idea is that it draws a line between the two divertor points
    and parametrizes it so that 0 is the left point and r is the right point
    and then outputs values that correspond to positions along that line

    If values are given that are outside 0 or 1 it extends beyond the point allowing
    the divertor plate to extend beyond the "island"
"""    
function write_classical_divertor(wf, rl, rr, zl, zr, phi;
                  left_extension=0.05, right_extension=0.05, closed=0.0)

    #duplicated code here should refactor at some point
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
        #get phi in degrees
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
            # Computes right and left side points, and the midpoint between 
            # them, and then writes them as (right, midpoint, left)
            rback, zback = close_divertor(rdiv[1], rdiv[end], zdiv[1], zdiv[end], closed)
            for k in 1:3
                write(wf, string(rback[k]*100))
                write(wf, "  ")
                write(wf, string(zback[k]*100))
                write(wf, "\n")
            end
            # Add the first point again to make a closed surface
            write(wf, string(rdiv[1]*100))
            write(wf, "  ")
            write(wf, string(zdiv[1]*100))
            write(wf, "\n")
        end
    end
end

"""
    make a classical from a json input file

    the json inputs are listed below
    `mgrid_file`: name of the mgrid file
    `island_boundary_file`: name of the island boundary file (csv format)
    `o_points_file`: name of the o_points file (csv format)
    `write_div_name`: name of the file to output to
    `write_div_tag`: tag to put on the top of the divertor output file
    `island`: number island in which to make the divertor
    `n_islands`: number of islands on the boundary
    `direction`: whether to move forward or backwards in phi
    `target`: the target value in degrees that the field line should intersect the divertor plate
    `npoints`: number of points in the divertor
    `reach_target`: How many "steps" it should take before you try to hit the target field intercept angle (default 1)
    `eval_direction`: either 1 or -1, a value that helps determine which direction to move the divertor either in
        or out of the island. May need verification to determine it has been set correctly
    `start_phi_deg`: where the divertor starts in toroidal angle
    `r1` and `z1`: the r and z points for the first point when going around CCW
    `r2` and `z2`: the r and z points for the second point when going around CCW
    `closed`: default 0.0, i.e. no thickness.  If true makes a closed divertor with wings at the target thickness

"""
function make_classical_divertor_from_json(json_input_file; verbose=true)
    json_data = JSON3.read(json_input_file)

    #check if this is an h5 i.e. HINT eq
    if json_data.mgrid_file[end-2:end] == ".h5"
        mg = read_hint_field(json_data.mgrid_file)
    else
        #make the mgrid
        mg = read_mgrid(json_data.mgrid_file, [1.0])
    end


    # Check for closed flag, which is important for EMC3-EIRENE calculations, 
    # which requires a closed surface
    if hasproperty(json_data, :closed)
        closed = json_data.closed
    else
        closed = 0.0
    end

    #check for optional extension flags
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

    #get the island boundary and o_points
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
        npol = "9" # + 3 points + 1 extra repeat of a corner to close divertor
    else
        npol = "5"
    end
    write(wf, string(ntor)*" "*npol*" "*string(mg.nfp)*" 0.0 0.0\n")

    #write_divertor(wf, brl, brr, bzl, bzr, bphi)
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
    """
    Creates a dictionary with keys the degrees from a divertor file and values
    the r and z values associated with that angle.
    """
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
        """
        Function to extract r and z from a Cartesian SVector.
        """
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

    # For every single point
    for i in 1:9
        # Calculate point in cartesian
        c = Cylindrical(r[i], phi, z[i])
        c_cart = CartesianFromCylindrical()(c)
        # Define vector from midpt to divertor point
        vec = midpt - c_cart
        # Move the point really close to midpt
        c_new = c_cart + vec*0.99 
        r_new, z_new = get_rz(c_new)
        rcap[i] = r_new
        zcap[i] = z_new
    end
    # Place these points on the new phi plane
    phicap = phi*(1 + istep)

    # Write the divertor points to a file
    # First, we write the old points, then we write the new points
    wf = open(str*".txt","w")
    write(wf, "#"*str*"\n")
    ntor = 2
    npol = 9 # Hardcoded, but should be extracted from the old file
    write(wf, string(ntor)*" "*string(npol)*" "*string(4)*" 0.0 0.0\n") # 4 is mg.nfp
    
    # Now I decide whether to write the cap first or last. 
    function write_points(r, phi, z)
        """
        Writing the divertor points.
        """
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
    # For the fwd divertor, the new points should be at the end, but for the bwd
    # divertor, the new points should be first
    if istep > 0
        write_points(r, phi, z)
        write_points(rcap, phicap, zcap)
    else
        write_points(rcap, phicap, zcap)
        write_points(r, phi, z)
    end

    close(wf)
end
