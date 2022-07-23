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

struct Diag
    d
end
Base.:*(D::Diag, u::AbstractVector) = Diagonal(D.d) * u

function update_D!(D::Diag, newval)
    copy!(D.d, newval)

    D
end

function ChainRulesCore.rrule(::typeof(update_D!), D, newval)

    function update_D!_pullback(dD)
        println("in update_D!_pullback")

        NoTangent(), dD, copy(dD.d)
    end

    update_D!(D, newval), update_D!_pullback
end

N = 8
u = rand(N)
p = rand(N)

D = Diag(zeros(N))

function loss(p; M=D)

    M = update_D!(M, p)

    sum(M * u)
end

grad = Zygote.gradient(loss, p)
