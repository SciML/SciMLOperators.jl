## Demonstration of Operator Algebras and Kron

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