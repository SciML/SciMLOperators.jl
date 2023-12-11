# Wrap a Fourier transform with SciMLOperators

In this tutorial, we will wrap a Fast Fourier Transform (FFT) in a SciMLOperator via the
`FunctionOperator` interface. FFTs are commonly used algorithms for performing numerical
interpolation and differentiation. In this example, we will use the FFT to compute the
derivative of a function.

## Copy-Paste Code

```
using SciMLOperators
using LinearAlgebra, FFTW

n = 256
L = 2π

dx = L / n
x  = range(start=-L/2, stop=L/2-dx, length=n) |> Array
u  = @. sin(5x)cos(7x);
du = @. 5cos(5x)cos(7x) - 7sin(5x)sin(7x);

k = rfftfreq(n, 2π*n/L) |> Array
m = length(k)
P = plan_rfft(x)

fwd(u, p, t) = P * u
bwd(u, p, t) = P \ u

fwd(du, u, p, t) = mul!(du, P, u)
bwd(du, u, p, t) = ldiv!(du, P, u)

F = FunctionOperator(fwd, x, im*k;
        T=ComplexF64,

        op_adjoint = bwd,
        op_inverse = bwd,
        op_adjoint_inverse = fwd,

        islinear=true,
        )

ik = im * DiagonalOperator(k)
Dx = F \ ik * F

Dx = cache_operator(Dx, x)

@show ≈(Dx * u, du; atol=1e-8)
@show ≈(mul!(copy(u), Dx, u), du; atol=1e-8)
```

## Explanation

We load `SciMLOperators`, `LinearAlgebra`, and `FFTW` (short for Fastest Fourier Transform
in the West), a common Fourier transform library. Next, we define an equispaced grid from
-π to π, and write the function `u` that we intend to differentiate. Since this is a
trivial example, we already know the derivative, `du`, and write it down to later test our
FFT wrapper.

```
using SciMLOperators
using LinearAlgebra, FFTW

L  = 2π
n  = 256
dx = L / n
x  = range(start=-L/2, stop=L/2-dx, length=n) |> Array

u  = @. sin(5x)cos(7x);
du = @. 5cos(5x)cos(7x) - 7sin(5x)sin(7x);

```

Now, we define the Fourier transform. Since our input is purely Real, we use the real
Fast Fourier Transform. The function `plan_rfft` outputs a real fast Fourier transform
object that can be applied to inputs that are like `x` as follows: `xhat = transform * x`,
and `LinearAlgebra.mul!(xhat, transform, x)`.  We also get `k`, the frequency modes sampled by
our finite grid, via the function `rfftfreq`.

```
k  = rfftfreq(n, 2π*n/L) |> Array
m  = length(k)
tr = plan_rfft(x)
```

Now we are ready to define our wrapper for the FFT object. To `FunctionOperator`, we
pass the in-place forward application of the transform,
`(du,u,p,t) -> mul!(du, transform, u)`, its inverse application,
`(du,u,p,t) -> ldiv!(du, transform, u)`, as well as input and output prototype vectors.

```
fwd(u, p, t) = P * u
bwd(u, p, t) = P \ u

fwd(du, u, p, t) = mul!(du, P, u)
bwd(du, u, p, t) = ldiv!(du, P, u)
F = FunctionOperator(fwd, x, im*k;
        T=ComplexF64,

        op_adjoint = bwd,
        op_inverse = bwd,
        op_adjoint_inverse = fwd,

        islinear=true,
        )
```

After wrapping the FFT with `FunctionOperator`, we are ready to compose it with other
SciMLOperators. Below, we form the derivative operator, and cache it via the function
`cache_operator` that requires an input prototype. We can test our derivative operator
both in-place, and out-of-place by comparing its output to the analytical derivative.

```
ik = im * DiagonalOperator(k)
Dx = F \ ik * F

@show ≈(Dx * u, du; atol=1e-8)
@show ≈(mul!(copy(u), Dx, u), du; atol=1e-8)
```

```
≈(Dx * u, du; atol = 1.0e-8) = true
≈(mul!(copy(u), Dx, u), du; atol = 1.0e-8) = true
```
