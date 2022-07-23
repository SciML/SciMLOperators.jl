#
using SciMLOperators
let
    # add test dependencies to env stack
    pkgpath = dirname(dirname(pathof(SciMLOperators)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using Test, Random, LinearAlgebra

using SciMLOperators.ChainRulesCore, ChainRulesTestUtils
using Zygote

Random.seed!(0)
N = 8
t = 0.0

f0 = (M, u) -> sum(M * u);
f1 = (M,u,p,t) -> sum(M(u,p,t))

## incorrect
#f2 = function(M,u,p,t)
#    ChainRulesCore.ignore_derivatives() do
#        SciMLOperators.update_coefficients!(M,u,p,t)
#    end
#    sum(M * u)
#end

update!(v, u) = copy!(v,u)
function ChainRulesCore.rrule(::typeof(update!), u::AbstractArray, v::AbstractArray)
    update!(v, u), (dv) -> (NoTangent(), copy(dv))
end

# IDEAS
# - move update_coefficients! to Zygote.buffer
# - figure out how gradients are accumulated at u, p, t
# - use adjoint operator (keep in min Affine will need special handling)
# - have rrule for update_func

#@testset "Matrix Operators" begin
    u = rand(N)
    p = rand(N)

    M = MatrixOperator(rand(N,N))
    D = DiagonalOperator(rand(N))
    A = AffineOperator(rand(N,N), rand(N,N), rand(N))

    @test Zygote.gradient(f0, M, u)[1].A isa Matrix
    @test Zygote.gradient(f0, D, u)[1].A isa Diagonal
    Zygote.gradient(f0, A, u)

    ## updates

    DD = DiagonalOperator(rand(N); update_func=(diag,u,p,t)->update!(diag,p))
    gg = Zygote.gradient(f1, DD, u, p, t)

#end

#@testset "Function Operator" begin
#    u = rand(N)
#
#    A = rand(N,N) |> Symmetric
#    F = lu(A)
#
#    fwd(du, u, p, t) = mul!(du, A, u)
#    bwd(du, u, p, t) = ldiv!(du, F, u)
#
#    # in place
#    F = FunctionOperator(
#                         fwd;
#
#                         isinplace=true,
#                         T=Float64,
#                         size=(N,N),
#
#                         input_prototype=u,
#                         output_prototype=A*u,
#
#                         op_inverse=bwd,
#
#                         opnorm=true,
#                         issymmetric=true,
#                         ishermitian=true,
#                         isposdef=true,
#                        )
#
#    grad = Zygote.gradient(f0, F, u)
##   test_rrule(*, F, u)
#end
#
