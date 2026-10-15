#=
3D Euler Characteristic Profile over the (cone, dir_long, dir_norm) filtration.

The 3D counterpart of `ecp2d.jl`, and very close to the elasticity repo's
`directional_ecp.jl` — this is the one stage of the pipeline that ports almost
unchanged, because `Ecp.compute_contributions_3d` already takes exactly the
(Z, Y, X, n_params) array `filtration3d.py` writes.

Two changes from the elasticity original:

  * **Skip work that is already done.** `isfile(out) ||` makes a nine-direction
    run over 5000 structures restartable; the original recomputes everything.
  * `n_params` is read from the array rather than assumed to be 3, so a
    2-channel filtration (`filtration3d.py --channels 2`) works without editing
    this file.  As in 2D, `vectorize_ecp` needs a **tuple** of steps, one per
    filtration parameter — a scalar raises `BoundsError`.

With 3 channels and --grid-res 6 the output is (6+1)^3 = 343 values, matching
METHODS.md; --grid-res 8 gives 729.

Usage:
  julia --project=<env> --threads auto scripts/ecp3d.jl <filt_dir> <out_dir> [--grid-res 6]
=#

using Glob
using Base.Threads
import NPZ
import Ecp

function get_ecp(input_file, output_file, grid_res)
    data = NPZ.npzread(input_file)
    n_params = size(data, 4)
    try
        contr = Ecp.compute_contributions_3d(data)
        out = Ecp.vectorize_ecp(contr, Tuple(grid_res for _ in 1:n_params))
        NPZ.npzwrite(output_file, vec(Float64.(out)))
    catch e
        println("Error in ECP for $input_file: ", e)
        NPZ.npzwrite(output_file, fill(0.0, (grid_res + 1)^n_params))
    end
end

function parse_args()
    if length(ARGS) < 2
        println("usage: julia scripts/ecp3d.jl <filtration_dir> <output_dir> [--grid-res N]")
        exit(1)
    end
    grid_res = 6
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
println("ECP: $(length(files)) files, grid-res $GRID_RES, $(nthreads()) threads")

done = Threads.Atomic{Int}(0)
@threads for i in eachindex(files)
    out = joinpath(OUTPUT_DIR, basename(files[i]))
    if !isfile(out)
        get_ecp(files[i], out, GRID_RES)
        Threads.atomic_add!(done, 1)
    end
end

println("ECP done: $(done[]) computed, $(length(files) - done[]) cached -> $OUTPUT_DIR")
