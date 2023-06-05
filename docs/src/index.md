# SciMLOperators.jl: Unified operator interface for `SciML.ai` and beyond

`SciMLOperators` is a package for managing linear, nonlinear,
time-dependent, and parameter dependent operators acting on vectors,
(or column-vectors of matrices). We provide wrappers for matrix-free
operators, fast tensor-product evaluations, pre-cached mutating
evaluations, as well as `Zygote`-compatible non-mutating evaluations.

The lazily implemented operator algebra allows the user to update the
operator state by passing in an update function that accepts arbirary
parameter objects. Further, our operators behave like `AbstractMatrix` types
thanks to  overloads defined for methods in `Base`, and `LinearAlgebra`.

Therefore, an `AbstractSciMLOperator` can be passed to `LinearSolve.jl`,
or `NonlinearSolve.jl` as a linear/nonlinear operator, or to
`OrdinaryDiffEq.jl` as an `ODEFunction`. Examples of usage within the
`SciML` ecosystem are provided in the documentation.

## Installation
`SciMLOperators.jl` is a registerd package and can be installed via

```
julia> import Pkg
julia> Pkg.add("SciMLOperators")
```

## Examples

Let `M`, `D`, `F` be matrix-based, diagonal-matrix-based, and function-based
`SciMLOperators` respectively.

```julia
N = 4
f = (u, p, t) -> u .* u

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
v = L1(u, p, t) # == L1 * u

u_kron = rand(N ^ 3)
v_kron = L3(u_kron, p, t) # == L3 * u_kron
```

For mutating operator evaluations, call `cache_operator` to generate
in-place cache so the operation is nonallocating.

```julia
α, β = rand(2)

# allocate cache
L2 = cache_operator(L2, u)
L4 = cache_operator(L4, u)

# allocation-free evaluation
L2(v, u, p, t) # == mul!(v, L2, u)
L4(v, u, p, t, α, β) # == mul!(v, L4, u, α, β)
```

The calling signature `L(u, p, t)`, for out-of-place evaluations is
equivalent to `L * u`, and the in-place evaluation `L(v, u, p, t, args...)`
is equivalent to `LinearAlgebra.mul!(v, L, u, args...)`, where the arguments
`p, t` are passed to `L` to update its state. More details are provided
in the operator update section below. While overloads to `Base.*`
and `LinearAlgebra.mul!` are available, where a `SciMLOperator` behaves
like an `AbstractMatrix`, we recommend sticking with the
`L(u, p, t)`, `L(v, u, p, t)`, `L(v, u, p, t, α, β)` calling signatures
as the latter internally update the operator state.

The `(u, p, t)` calling signature is standardized over the `SciML`
ecosystem and is flexible enough to support use cases such as time-evolution
in ODEs, as well as sensitivity computation with respect to the parameter
object `p`.

## Features

* Matrix-free operators with `FunctionOperator`
* Fast tensor product evaluation with `TensorProductOperator`
* Lazy algebra: addition, subtraction, multiplication, inverse, adjoint, transpose
* Couple fast methods for operator evaluation with inversion via `InvertibleOperator`
* One-line API to update operator state depending on arbitrary parameters.
* Mutating, nonmutating update behaviour (Zygote compatible)
* One-line API to pre-caching operators for in-place operator evaluations

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - On the Julia Discourse forums (look for the [modelingtoolkit tag](https://discourse.julialang.org/tag/modelingtoolkit)
    - See also [SciML Community page](https://sciml.ai/community/)

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
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the 
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
```
```@raw html
">project</a> file.
```
