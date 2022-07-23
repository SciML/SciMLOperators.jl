using SciMLOperators
let
    # add test dependencies to env stack
    pkgpath = dirname(dirname(pathof(SciMLOperators)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using Test, Random, LinearAlgebra
using SciMLOperators.ChainRulesCore, ChainRulesTestUtils, Zygote

Random.seed!(0)
N = 8

u = rand(N)
p = rand(N)
t = rand()

function up!(diag, u, p, t)
    copy!(diag, p)
end

function ChainRulesCore.rrule(::typeof(up!), diag, u, p, t)

    function up!_pullback(ddiag)
        println("chodu")
        NoTangent(), ddiag, ZeroTangent(), copy(ddiag), ZeroTangent()
    end

    up!(diag,u,p,t), up!_pullback
end

f1 = (M,u,p,t) -> sum(M(u,p,t))

function f2(M,u,p,t)

    p = Zygote.@showgrad(p)
    up!(D.A.diag, u, p, t) # <-- pullback not being called

    sum(D * u)
end

D  = DiagonalOperator(rand(N); update_func=up!)
#g1 = Zygote.gradient(f1, D, u, p, t); @show g1[3]
g2 = Zygote.gradient(f2, D, u, p, t); @show g2[3]
