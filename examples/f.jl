# add dependencies to env stack
pkgpath = dirname(dirname(@__FILE__))
tstpath = joinpath(pkgpath, "test")
!(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)

using SciMLOperators, LinearAlgebra, FFTW, Test, Plots

n = 256
L = 2π

dx = L / n
x  = range(start=-L/2, stop=L/2-dx, length=n) |> Array

k   = rfftfreq(n, 2π*n/L) |> Array
tr  = plan_rfft(x)
itr = plan_irfft(im*k, n)

T = eltype(x)
ComplexT = if T isa Type{Float16}
    ComplexF16
elseif T isa Type{Float32}
    ComplexF32
else
    ComplexF64
end

tr_iip = FunctionOperator(
                          (du,u,p,t) -> mul!(du, tr, u);
                          isinplace=true,
                          T=ComplexT,
                          size=(length(k),n),

#                         op_adjoint = 
                          op_inverse = (du,u,p,t) -> ldiv!(du, tr, u)
                         )

tr_oop = FunctionOperator(
                          (u,p,t) -> tr * u;
                          isinplace=false,
                          T=ComplexT,
                          size=(length(k),n),

                          op_inverse = (u,p,t) -> tr \ u,
                         )

ik = im * DiagonalOperator(k)

D_iip = tr_iip \ ik * tr_iip
D_oop = tr_oop \ ik * tr_oop

u = @. sin(5x)cos(7x);
du = @. 5cos(5x)cos(7x) - 7sin(5x)sin(7x);

D = D_oop
#D = D_iip

v = D * u

@test ≈(v, du; atol=1e-8)
#
