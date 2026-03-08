using LinearAlgebra

"""
    run_micro_grid_search(problem, xp, Nbasis, alpha_start, step_size; search_radius=2)

A brute-force micro-grid search that evaluates adjacent "rational terraces" directly, bypassing Nelder-Mead.
Since EVAL_ALPHA_TOL creates flat regions, continuous optimization methods can stall.
This evaluates a discrete grid by stepping `search_radius` ticks of `step_size` in each search direction.

Best used when `length(alpha_start)` (nfree) is relatively small (e.g., 1 to 4).
"""
function run_micro_grid_search(
    problem, 
    xp::Vector{Rational{BigInt}}, 
    Nbasis::Matrix{Rational{BigInt}}, 
    alpha_start::Vector{Float64}, 
    trace_path::String;
    step_size::Float64 = parse(Float64, get(ENV, "SBP8_EVAL_ALPHA_TOL", "1e-8")),
    search_radius::Int = 3
)
    nfree = size(Nbasis, 2)
    最佳_obj = Inf
    best_alpha = copy(alpha_start)
    best_res = nothing

    trace_io = open(trace_path, "w")
    println(trace_io, "eval\tobj\tmax_real\tmax_imag\tv_full_pd\tstatus\treason")

    # Evaluate the anchor first
    eval_count = 0
    function eval_point(alpha::Vector{Float64})
        eval_count += 1
        res = evaluate_alpha_for_opt(problem, xp, Nbasis, alpha)
        
        if res.obj < 最佳_obj
            最佳_obj = res.obj
            best_alpha = copy(alpha)
            best_res = res
        end
        
        println(trace_io, "$eval_count\t$(res.obj)\t$(res.max_real)\t$(res.max_imag)\t$(res.v_full_pd)\t$(res.status)\t$(res.reason)")
        if eval_count % 25 == 0
            flush(trace_io)
        end
        return res
    end
    
    println("Starting fallback Micro-Grid Search centered at anchor...")
    println("Anchor       : ", alpha_start)
    println("Step Size    : ", step_size)
    println("Search Radius: +/-", search_radius)
    
    # Base Evaluation
    best_res = eval_point(alpha_start)
    最佳_obj = best_res.obj

    # If nfree is manageable (e.g., <= 4), do a full cartesian grid
    if nfree <= 4
        ranges = [(-search_radius:search_radius) for _ in 1:nfree]
        # CartesianProduct logic using Iterators
        grid = Iterators.product(ranges...)
        
        for offsets in grid
            all(iszero, offsets) && continue # Skip the center since we already evaluated it
            
            # Form grid point
            current_alpha = alpha_start .+ (collect(offsets) .* step_size)
            eval_point(current_alpha)
        end
    else
        # For larger bases, full grid is too large. Fallback to discrete coordinate descent
        println("Warning: nfree = $nfree is too large for full grid. Using discrete coordinate descent.")
        improved = true
        while improved
            improved = false
            for d in 1:nfree
                for offset in [-step_size, step_size]
                    current_alpha = copy(best_alpha)
                    current_alpha[d] += offset
                    res = eval_point(current_alpha)
                    if res.obj < 最佳_obj
                        improved = true
                        # Note: 最佳_obj and best_alpha are updated inside eval_point
                    end
                end
            end
        end
    end

    close(trace_io)
    println("Grid search completed (`$eval_count` grid points evaluated).")
    
    return (
        alpha_opt = best_alpha,
        res_opt = best_res,
        best_alpha = best_alpha,
        best_res = best_res,
        eval_count = eval_count
    )
end
