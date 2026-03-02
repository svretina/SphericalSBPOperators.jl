function _gundlach_construct_mass_data(r,
                                       Geven,
                                       S_seed_diag,
                                       accuracy_order::Int,
                                       p::Int,
                                       Nh::Int;
                                       boundary_count::Union{Nothing, Int} = nothing,
                                       mass_solver::Symbol = :seed,
                                       mass_solver_opts::NamedTuple = (;),
                                       seed_banded::Bool = true,
                                       seed_band_scale::Real = 1 // 10^12,
                                       snap_factor::Float64 = 64.0)
    if mass_solver === :seed
        if seed_banded
            seed_v_offdiag = _split_mass_seed_offdiag(
                                                       S_seed_diag,
                                                       accuracy_order,
                                                       :collocated;
                                                       band_scale = seed_band_scale
                                                      )
            seed_regularization = get(mass_solver_opts, :seed_regularization, 0.0)
            seed_pd_epsilon = get(mass_solver_opts, :seed_pd_epsilon, Float64(seed_band_scale))
            seed_farfield_points = get(mass_solver_opts, :seed_farfield_points, min(2, Nh))
            return construct_split_mass_matrices_seed(
                                                      r,
                                                      Geven;
                                                      accuracy_order = accuracy_order,
                                                      p = p,
                                                      layout = :collocated,
                                                      boundary_count = boundary_count,
                                                      normalization_index = Nh,
                                                      farfield_points = seed_farfield_points,
                                                      s_seed = S_seed_diag,
                                                      v_seed = S_seed_diag,
                                                      v_offdiag_seed = seed_v_offdiag,
                                                      regularization = seed_regularization,
                                                      pd_epsilon = seed_pd_epsilon,
                                                      strict = true,
                                                      snap_factor = snap_factor
                                                     )
        end

        return construct_split_mass_matrices(
                                             r;
                                             accuracy_order = accuracy_order,
                                             p = p,
                                             layout = :collocated,
                                             s_diag = S_seed_diag,
                                             v_diag = S_seed_diag,
                                             strict = true,
                                             snap_factor = snap_factor
                                            )
    elseif mass_solver === :optimization
        seed_v_offdiag = _split_mass_seed_offdiag(
                                                   S_seed_diag,
                                                   accuracy_order,
                                                   :collocated;
                                                   band_scale = seed_band_scale
                                                  )
        optimization_regularization = get(mass_solver_opts, :optimization_regularization, 0.0)
        optimization_pd_epsilon = get(mass_solver_opts, :optimization_pd_epsilon, Float64(seed_band_scale))
        optimization_farfield_points = get(mass_solver_opts, :optimization_farfield_points, min(2, Nh))
        optimization_max_iter = get(mass_solver_opts, :optimization_max_iter, 5_000)
        optimization_tolerance = get(mass_solver_opts, :optimization_tolerance, 1.0e-10)
        optimization_print_level = get(mass_solver_opts, :optimization_print_level, 0)
        return construct_split_mass_matrices_optimization(
                                                          r,
                                                          Geven;
                                                          accuracy_order = accuracy_order,
                                                          p = p,
                                                          layout = :collocated,
                                                          boundary_count = boundary_count,
                                                          normalization_index = Nh,
                                                          farfield_points = optimization_farfield_points,
                                                          s_seed = S_seed_diag,
                                                          v_seed = S_seed_diag,
                                                          v_offdiag_seed = seed_v_offdiag,
                                                          regularization = optimization_regularization,
                                                          pd_epsilon = optimization_pd_epsilon,
                                                          strict = true,
                                                          snap_factor = snap_factor,
                                                          max_iter = optimization_max_iter,
                                                          tolerance = optimization_tolerance,
                                                          print_level = optimization_print_level
                                                         )
    elseif mass_solver === :cddlib
        return construct_split_mass_matrices_cddlib(
                                                    r,
                                                    Geven;
                                                    accuracy_order = accuracy_order,
                                                    p = p,
                                                    layout = :collocated,
                                                    boundary_count = boundary_count,
                                                    normalization_index = Nh,
                                                    farfield_points = min(2, Nh),
                                                    mass_solver_opts...
                                                   )
    else
        throw(ArgumentError("`mass_solver` must be :seed, :optimization, or :cddlib; got `$mass_solver`."))
    end
end

@inline function _gundlach_skip_gegeven_repair(mass_solver::Symbol, seed_banded::Bool)
    return (mass_solver === :seed && seed_banded) || mass_solver === :optimization
end
