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

function update!(oldval, newval)
    copy!(oldval, newval)
end

function ChainRulesCore.rrule(::typeof(update!), oldval, newval)

    function update!_pullback(dnew)
        println("in update!_pullback")

        NoTangent(), dnew, copy(dnew)
    end

    update!(oldval, newval), update!_pullback
end

N = 8
u = rand(N)
p = rand(N)

D = Diag(zeros(N))

function loss(p; M=D)

    update!(M.d, p)

    sum(M * u)
end

grad = Zygote.gradient(loss, p)
