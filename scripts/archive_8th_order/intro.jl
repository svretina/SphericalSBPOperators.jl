using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004

source = MattssonNordström2004()
ops = spherical_operators(source;
                          accuracy_order = 6,
                          N = 32,
                          R = 1.0,
                          p = 2)
report = validate(ops; verbose = true)

println("\nConstructed folded spherical operators with Nh = $(ops.Nh).")
println("SBP residual (excluding origin row): $(report.sbp.sbp_no_origin)")
