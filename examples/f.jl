# add dependencies to env stack
pkgpath = dirname(dirname(@__FILE__))
tstpath = joinpath(pkgpath, "test")
!(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)

using SciMLOperators, FFTW, Test

n = 32
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

transform = FunctionOperator(
                             (du,u,p,t) -> mul!(du, tr, u);
                             isinplace=true,
                             T=ComplexT,
                             size=(length(k),n),

#                            op_adjoint = 
                             op_inverse = (du,u,p,t) -> ldiv!(du, tr, u)
                            )

ik = im * DiagonalOperator(k)
D = transform \ ik * transform

u0 = @. sin(5x); du0 = @. 5cos(5x);
u1 = @. exp(2x); du0 = @. 2u1

@test D * u0 ≈ du0
@test D * u1 ≈ du1

