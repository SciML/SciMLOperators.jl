using SciMLOperators, JET, Test
using SciMLTesting: run_qa
# Load the weak-dependency triggers so the package extensions are loaded and
# analyzed by the ExplicitImports checks below.
using LoopVectorization, SparseArrays, StaticArraysCore

# Names whose dependency has NOT marked them public but which are genuinely
# required as qualified accesses:
#   - Base.Cartesian `@nexprs`/`@ntuple` macros and `Base.promote_eltype`
#     (no public equivalent)
#   - `LinearAlgebra.{Adjoint,Transpose}Factorization` wrapper types used for
#     adjoint/transpose factorization dispatch (no public name)
#   - `Base.Broadcast.Broadcasted` / `StaticArraysCore.StaticArrayStyle` in
#     the StaticArraysCore extension's `copyto!` signature
#   - SciMLOperators' own private `_has_tensor_outer_mul_fast` /
#     `_tensor_outer_mul_fast!` hooks, extended by the LoopVectorization
#     extension (deliberately internal, underscore-prefixed)
qualified_access_ignore = (
    :var"@nexprs", :var"@ntuple", :promote_eltype,
    :AdjointFactorization, :TransposeFactorization,
    :Broadcasted, :StaticArrayStyle,
    :_has_tensor_outer_mul_fast, :_tensor_outer_mul_fast!,
)

# `README` is a DocStringExtensions abbreviation used only in the module-level
# docstring (`$(README)` at the top of src/SciMLOperators.jl). It is resolved
# eagerly at precompile time, so the explicit import is load-bearing, but
# ExplicitImports does not count module-docstring interpolation as a use and so
# reports it as stale. Ignore that single false positive.
#
# Aqua `ambiguities` (24 found) and `unbound_args` (FunctionOperator inner ctor
# unbound `N`, src/func.jl:303) currently fail; disabled via `aqua_kwargs`. JET
# reports 3 possible errors (opnorm/resize!/ldiv!); run with `broken = true`.
# All tracked in https://github.com/SciML/SciMLOperators.jl/issues/392.
run_qa(
    SciMLOperators;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = false, unbound_args = false),
    jet_kwargs = (; target_defined_modules = true, broken = true),
    ei_kwargs = (;
        no_stale_explicit_imports = (; ignore = (:README,)),
        all_qualified_accesses_are_public = (; ignore = qualified_access_ignore),
    ),
)
