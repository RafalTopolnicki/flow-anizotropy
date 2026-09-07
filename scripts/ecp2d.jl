#=
2D Euler Characteristic Profile over the (wedge, dir) bifiltration.

The 2D counterpart of `direction-aware-tda/scripts/directional_ecp.jl`.
Two changes from the 3D original:

  * `Ecp.compute_contributions_2d` instead of `_3d` — it takes exactly the
    (X, Y, n_params) array `filtration2d.py` writes.
  * `vectorize_ecp` needs a **tuple** of steps, one per filtration parameter.
    Passing a scalar (as the shape of the 3D call might suggest) raises
    `BoundsError: attempt to access Int64 at index [2]`.

With 2 channels and --grid-res g the output is (g+1)^2 values, e.g. 81 at
g = 8, against 343 for the 3-channel 3D filtration.

Usage:
  julia --project=<env> scripts/ecp2d.jl <filtration_dir> <output_dir> [--grid-res 8]
=#

using Glob
using Base.Threads
import NPZ
import Ecp

function get_ecp(input_file, output_file, grid_res)
    data = NPZ.npzread(input_file)
    n_params = size(data, 3)
    try
        contr = Ecp.compute_contributions_2d(data)
        out = Ecp.vectorize_ecp(contr, Tuple(grid_res for _ in 1:n_params))
        NPZ.npzwrite(output_file, vec(Float64.(out)))
    catch e
        println("Error in ECP for $input_file: ", e)
        NPZ.npzwrite(output_file, fill(0.0, (grid_res + 1)^n_params))
    end
end

function parse_args()
    if length(ARGS) < 2
        println("usage: julia scripts/ecp2d.jl <filtration_dir> <output_dir> [--grid-res N]")
        exit(1)
    end
    grid_res = 8
    i = 3
    while i <= length(ARGS)
        if ARGS[i] == "--grid-res" && i + 1 <= length(ARGS)
            grid_res = parse(Int, ARGS[i + 1]); i += 2
        else
            println("unknown argument: $(ARGS[i])"); exit(1)
        end
    end
    return ARGS[1], ARGS[2], grid_res
end

const INPUT_DIR, OUTPUT_DIR, GRID_RES = parse_args()

mkpath(OUTPUT_DIR)
files = glob("*.npy", INPUT_DIR)
println("ECP: $(length(files)) files, grid-res $GRID_RES -> $((GRID_RES + 1)^2) features, $(nthreads()) threads")

@threads for i in eachindex(files)
    out = joinpath(OUTPUT_DIR, basename(files[i]))
    isfile(out) || get_ecp(files[i], out, GRID_RES)
end

println("ECP done -> $OUTPUT_DIR")
