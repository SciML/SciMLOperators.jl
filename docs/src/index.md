# SciMLOperators.jl: Unified operator interface for `SciML.ai` and beyond

`SciMLOperators` is a package for managing linear, nonlinear,
time-dependent, and parameter dependent operators acting on vectors,
(or column-vectors of matrices). We provide wrappers for matrix-free
operators, fast tensor-product evaluations, pre-cached mutating
evaluations, as well as `Zygote`-compatible non-mutating evaluations.

The lazily implemented operator algebra allows the user to update the
operator state by passing in an update function that accepts arbitrary
parameter objects. Further, our operators behave like `AbstractMatrix` types
thanks to overloads defined for methods in `Base`, and `LinearAlgebra`.

Therefore, an `AbstractSciMLOperator` can be passed to `LinearSolve.jl`,
or `NonlinearSolve.jl` as a linear or nonlinear operator, or to
`OrdinaryDiffEq.jl` as an `ODEFunction`. Examples of usage within the
`SciML` ecosystem are provided in the documentation.

## Installation

To install SciMLOperators.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("SciMLOperators")
```

## Examples

Let `M`, `D`, `F` be matrix-based, diagonal-matrix-based, and function-based
`SciMLOperators` respectively.

```julia
N = 4
f = (v, u, p, t) -> u .* v

M = MatrixOperator(rand(N, N))
D = DiagonalOperator(rand(N))
F = FunctionOperator(f, zeros(N), zeros(N))
```

Then, the following codes just work.

```julia
L1 = 2M + 3F + LinearAlgebra.I + rand(N, N)
L2 = D * F * M'
L3 = kron(M, D, F)
L4 = M \ D
L5 = [M; D]' * [M F; F D] * [F; D]
```

Each `L#` can be applied to `AbstractVector`s of appropriate sizes:

```julia
p = nothing # parameter struct
t = 0.0     # time

u = rand(N)
v = rand(N)
w = L1(v, u, p, t) # == L1 * v

v_kron = rand(N^3)
w_kron = L3(v_kron, u, p, t) # == L3 * v_kron
```

For mutating operator evaluations, call `cache_operator` to generate an
in-place cache, so the operation is nonallocating.

```julia
α, β = rand(2)

# allocate cache
L2 = cache_operator(L2, u)
L4 = cache_operator(L4, u)

# allocation-free evaluation
L2(w, v, u, p, t) # == mul!(w, L2, v)
L4(w, v, u, p, t, α, β) # == mul!(w, L4, v, α, β)
```

The calling signature `L(v, u, p, t)`, for out-of-place evaluations, is
equivalent to `L * v`, and the in-place evaluation `L(w, v, u, p, t, args...)`
is equivalent to `LinearAlgebra.mul!(w, L, v, args...)`, where the arguments
`u, p, t` are passed to `L` to update its state. More details are provided
in the operator update section below.

The `(v, u, p, t)` calling signature is standardized over the `SciML`
ecosystem and is flexible enough to support use cases such as time-evolution
in ODEs, as well as sensitivity computation with respect to the parameter
object `p`.

## Features

  - Matrix-free operators with `FunctionOperator`
  - Fast tensor product evaluation with `TensorProductOperator`
  - Lazy algebra: addition, subtraction, multiplication, inverse, adjoint, and transpose
  - Couple fast methods for operator evaluation with inversion via `InvertibleOperator`
  - One-line API to update operator state depending on arbitrary parameters.
  - Mutating and nonmutating update behavior (Zygote compatible)
  - One-line API for pre-caching operators for in-place operator evaluations

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
