# Graph Report - .  (2026-05-26)

## Corpus Check
- 81 files · ~157,790 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1044 nodes · 2284 edges · 54 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]

## God Nodes (most connected - your core abstractions)
1. `SummationByPartsOperators` - 26 edges
2. `SphericalSBPOperators` - 19 edges
3. `plot_all_resolution_analytic_errors_for_paper()` - 19 edges
4. `main()` - 19 edges
5. `plot_all_resolution_analytic_snapshots_for_paper()` - 18 edges
6. `LinearAlgebra` - 17 edges
7. `_gundlach_pointwise_error_history()` - 17 edges
8. `SphericalSBPOperators` - 16 edges
9. `SparseArrays` - 16 edges
10. `diagnose()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `spherical_operators()` --calls--> `_infer_interior_accuracy_order()`  [EXTRACTED]
  src/diagonal_mass_mixed_order/construct.jl → src/diagonal_mass/construct.jl
- `diagnose()` --calls--> `_tofloat_vec()`  [EXTRACTED]
  src/staggered/diagnostics.jl → src/diagonal_mass/diagnostics.jl
- `diagnose()` --calls--> `_region_indices()`  [EXTRACTED]
  src/staggered/diagnostics.jl → src/diagonal_mass/diagnostics.jl
- `diagnose()` --calls--> `_table_rows()`  [EXTRACTED]
  src/staggered/diagnostics.jl → src/diagonal_mass/diagnostics.jl
- `diagnose()` --calls--> `_print_table()`  [EXTRACTED]
  src/staggered/diagnostics.jl → src/diagonal_mass/diagnostics.jl

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (121): load_simulation_files(), make_all_paper_plots(), _paper_all_families_first_point_error_filename(), _paper_all_families_first_point_error_title(), _paper_all_family_first_point_divergence_error_data(), _paper_all_resolution_analytic_error_figure(), _paper_all_resolution_analytic_snapshot_figure(), _paper_all_resolution_error_snapshot_figure() (+113 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (82): animate_field_evolution, CairoMakie, _construct_operator(), construct_operators(), _normalized_families(), _print_summary(), _validate_operator(), DelimitedFiles (+74 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (55): _advance_to_time_index(), _align_reference_to_grid(), _analytic_reference_from_kind_summary(), _analytic_reference_from_metadata(), _canonical_gundlach_variable(), _canonical_wave_field(), _coerce_bundle_metadata(), _common_time_sampling() (+47 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (48): Dates, _algorithm_tag(), _boundary_group(), _build_characteristic_data(), _build_even_gaussian_pi_data(), _build_even_gaussian_pi_zero_psi_data(), _build_initial_data(), _build_regular_left_moving_spherical_gaussian_data() (+40 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (49): _adaptive_tol_study(), benchmark_wave_integrators(), _bigfloat_reflecting_rate(), _boundary_flux_term(), _build_saveat(), _check_provided_dt_stability(), _classify_energy_rate(), compute_stability_limit() (+41 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (33): _assemble_constraints(), _canonical_grid(), DiagMassQPConfig, _independent_row_indices(), _infer_boundary_accuracy(), _infer_boundary_count(), _integral_rhs(), _next_cell_volume_from_origin() (+25 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (25): _as_dense_square_matrix(), assemble_block_operator(), _build_even_folding_maps(), build_folded_sbp_dissipation(), discrete_energy(), dominant_real_projection(), inward_packet_initial_data(), _normalize_boundary_condition() (+17 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (30): Random, add_fix_col!(), build_problem(), build_step_system(), choose_pairs(), col_label(), dense_candidate_cols(), dense_pairs() (+22 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (20): apply_divergence(), apply_even_gradient(), apply_odd_derivative(), _build_non_diagonal_ops(), check_odd(), _default_scale_eltype(), enforce_odd!(), _exact_rank() (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (28): Optim, add_fix_col!(), build_matrices(), build_problem(), compose_x(), enforce_exact_residual!(), ensure_dir(), evaluate_alpha_for_opt() (+20 more)

### Community 10 - "Community 10"
Cohesion: 0.14
Nodes (25): add_fix_col!(), build_problem(), col_label(), dense_candidate_cols(), dense_pairs(), ensure_dir(), evaluate_solution(), extract_pairs() (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (27): add_fix_col!(), better_candidate(), build_matrices(), build_problem(), col_label(), compose_x(), enforce_exact_residual!(), ensure_dir() (+19 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (28): add_fix_col!(), base_zero_pairs(), build_problem(), build_specs(), candidate_columns(), coarse_step_for_col(), col_label(), default_center_for_col() (+20 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (27): _as_big_rational(), _assemble_hyperbolic_block(), _boundary_model_tag(), _build_banded_ops(), _clean_small(), collect_sbp_sources(), _compute_order_rows(), _construct_source_instance() (+19 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (19): apply_boundary_conditions!(), apply_characteristic_bc_sat!(), apply_symmetry_constraints!(), apply_symmetry_rhs!(), apply_symmetry_state!(), apply_wave_constraints!(), boundary_characteristic_residual(), boundary_characteristics() (+11 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (20): _axis_limits(), _collect_method_spectra(), _export_plots_to_dir(), generate_banded_spectrum_plots(), generate_diagonal_spectrum_plots(), generate_method_spectrum_plots(), _hyperbolic_plot_title(), _legend_entry() (+12 more)

### Community 16 - "Community 16"
Cohesion: 0.19
Nodes (21): InteractiveUtils, _build_spherical_ops(), _citation_entry(), collect_sbp_sources(), _construct_source_instance(), _fallback_bib_entry(), first_order_hyperbolic_spatial_operator(), first_order_hyperbolic_spectrum() (+13 more)

### Community 17 - "Community 17"
Cohesion: 0.2
Nodes (20): _analytic_profile(), _analytic_profile_divergence_exact(), _comparison_error_ylabel(), _comparison_filename(), _comparison_panel_title(), _comparison_profile_data(), _comparison_shared_top_row_axis(), _comparison_title() (+12 more)

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (18): _add_row!(), _assemble_mass(), BlockMassQPConfig, _build_block_pairs(), _build_linear_system(), _canonical_grid(), _independent_row_indices(), _infer_boundary_accuracy() (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.23
Nodes (14): assemble_block_operator(), discrete_energy(), dominant_real_projection(), inward_packet_initial_data(), _normalize_boundary_condition(), nullspace_and_checkerboard(), oscillation_metrics(), reflection_ibvp_test() (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.25
Nodes (16): _argmax_sparse_float(), _diag_and_offdiag_max_float(), diagnose(), interpret_diagnostics(), _maxabs_float(), _maxabs_sparse_float(), _maxabs_sparse_rows_float(), _pointwise_error_summary() (+8 more)

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (13): AbstractWaveOperatorKind, wave_D_kernel(), wave_D_mul!(), wave_Geven_kernel(), wave_Geven_mul!(), _wave_kernel_boundary_mul!(), _wave_kernel_interior_range(), wave_kernel_mul!() (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.32
Nodes (10): animate_field_evolution(), _cell_edges(), generate_simulation_plots(), plot_bc_energy_comparison(), plot_energy_history(), plot_field_evolution(), plot_initial_data(), _rescaled_fields() (+2 more)

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (3): apply_divergence(), apply_even_gradient(), apply_odd_derivative()

### Community 24 - "Community 24"
Cohesion: 0.36
Nodes (10): _even_cap(), _even_degrees(), _near_boundary_max(), _odd_cap(), _odd_degrees(), _relative_pattern(), _resolve_split_quadrature_orders(), _row_closure_flag() (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.36
Nodes (10): _sbp6_exp_constraint_rows(), sbp6_exp_construct_divergence(), _sbp6_exp_interior_diag_indices(), _sbp6_exp_left_diag_indices(), sbp6_exp_operators(), _sbp6_exp_right_diag_indices(), sbp6_exp_scalar_mass_gradient(), sbp6_exp_solve_accuracy_constraints() (+2 more)

### Community 26 - "Community 26"
Cohesion: 0.44
Nodes (9): _extract_laplacian_factors(), _hasprop(), laplacian_matrix(), laplacian_spectrum(), _negative_x_limits(), save_laplacian_spectrum_plot(), save_laplacian_spectrum_sources_plot(), _source_label() (+1 more)

### Community 27 - "Community 27"
Cohesion: 0.42
Nodes (9): _sbp4_as_big_rational(), _sbp4_constraint_error(), _sbp4_constraint_rows(), _sbp4_maxabs(), sbp4_operators(), sbp4_scalar_mass_gradient(), sbp4_solve_accuracy_constraints(), sbp4_v_offdiag_pairs() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.42
Nodes (9): _sbp6_as_big_rational(), _sbp6_constraint_error(), _sbp6_constraint_rows(), _sbp6_maxabs(), sbp6_operators(), sbp6_scalar_mass_gradient(), sbp6_solve_accuracy_constraints(), sbp6_v_offdiag_pairs() (+1 more)

### Community 29 - "Community 29"
Cohesion: 0.42
Nodes (9): _even_gaussian_pi_zero_psi_F(), _even_gaussian_pi_zero_psi_Fprime(), _even_gaussian_pi_zero_psi_integral_F(), _even_gaussian_pi_zero_psi_parameters(), _even_gaussian_pi_zero_psi_pi(), _even_gaussian_pi_zero_psi_primitive_F(), even_gaussian_pi_zero_psi_profile(), _even_gaussian_pi_zero_psi_psi() (+1 more)

### Community 30 - "Community 30"
Cohesion: 0.86
Nodes (8): _regular_left_moving_spherical_gaussian_parameters(), regular_left_moving_spherical_gaussian_phi_exact(), regular_left_moving_spherical_gaussian_pi_exact(), regular_left_moving_spherical_gaussian_profile(), regular_left_moving_spherical_gaussian_profile_prime(), regular_left_moving_spherical_gaussian_profile_second(), regular_left_moving_spherical_gaussian_psi_exact(), regular_left_moving_spherical_gaussian_solution()

### Community 31 - "Community 31"
Cohesion: 0.54
Nodes (7): _activate_cairo_backend(), mytheme_aps(), mytheme_aps_publication(), mytheme_aps_spectrum(), _with_aps_theme(), _with_publication_theme(), _with_spectrum_theme()

### Community 32 - "Community 32"
Cohesion: 0.29
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 0.67
Nodes (6): _has_origin_node(), _resolve_enforce_origin(), _wave_boundary_cache(), _wave_scalar_mass(), _wave_vector_mass(), WaveODEParams()

### Community 34 - "Community 34"
Cohesion: 0.52
Nodes (6): apply_boundary_conditions!(), apply_characteristic_bc_sat!(), apply_wave_constraints!(), boundary_characteristic_residual(), boundary_characteristics(), _normalize_boundary_condition()

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (2): characteristic_initial_data(), _profile_vector()

### Community 36 - "Community 36"
Cohesion: 0.6
Nodes (5): _high_precision_eigen(), _high_precision_schur_values(), _spectrum_as_big_rational(), _spectrum_float64x4_matrix(), _spectrum_rationalize_matrix()

### Community 37 - "Community 37"
Cohesion: 0.6
Nodes (5): _energy_plot_values(), _energy_tickformat(), _energy_ticklabel(), plot_wave_energy_histories(), plot_wave_energy_history()

### Community 38 - "Community 38"
Cohesion: 0.6
Nodes (4): _apply_cached_wave_rhs_constraints!(), wave_rhs!(), wave_system_ode!(), wave_system_ode_vec!()

### Community 39 - "Community 39"
Cohesion: 0.47
Nodes (3): apply_symmetry_constraints!(), apply_symmetry_state!(), initialize_wave_state!()

### Community 40 - "Community 40"
Cohesion: 0.67
Nodes (2): _boundary_closure_width_from_operator(), _build_full_grid_objects()

### Community 41 - "Community 41"
Cohesion: 0.83
Nodes (3): plot_wave_snapshot(), plot_wave_snapshot_resolutions(), _snapshot_series()

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (3): _analytic_boundary_state(), analytic_wave_solution(), _analytic_wave_solution_free()

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (1): _maxabs()

### Community 44 - "Community 44"
Cohesion: 0.67
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): AbstractSphericalOperators

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (0): 

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (0): 

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **13 isolated node(s):** `LinearSolve`, `SpecialFunctions`, `NonDiagonalMassSphericalOperators`, `MakiePublication`, `StaggeredSphericalOperators` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 45`** (2 nodes): `_sat_boundary_jacobian_entries()`, `sat_terms.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (2 nodes): `types.jl`, `AbstractSphericalOperators`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (2 nodes): `main()`, `run_block_mass_qp.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (2 nodes): `main()`, `run_diag_mass_qp.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `folding.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `folding.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Wave.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Jacobians.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `construct.jl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SummationByPartsOperators` connect `Community 1` to `Community 0`, `Community 3`, `Community 5`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 16`, `Community 18`, `Community 19`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `SphericalSBPOperators` connect `Community 1` to `Community 0`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 15`, `Community 16`, `Community 19`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `Printf` connect `Community 1` to `Community 0`, `Community 3`, `Community 5`, `Community 13`, `Community 15`, `Community 16`, `Community 18`, `Community 19`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **What connects `LinearSolve`, `SpecialFunctions`, `NonDiagonalMassSphericalOperators` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.03 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._