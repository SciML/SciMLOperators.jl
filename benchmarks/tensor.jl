using SciMLOperators, LinearAlgebra, BenchmarkTools
using SciMLOperators: IdentityOperator, ⊗

Id = IdentityOperator{12}()
A = rand(12,12)
B = rand(12,12)
C = rand(12,12)

println("#===============================#")
println("2D Tensor Products")
println("#===============================#")

println("⊗(A, B)")

u = rand(12^2, 100)
v = rand(12^2, 100)

T = ⊗(A, B)
T = cache_operator(T, u)

@btime mul!($v, $T, $u)

println("⊗(I, B)")

u = rand(12^2, 100)
v = rand(12^2, 100)

T = ⊗(Id, B)
T = cache_operator(T, u)

@btime mul!($v, $T, $u)

println("⊗(A, I)")

u = rand(12^2, 100)
v = rand(12^2, 100)

T = ⊗(A, Id)
T = cache_operator(T, u)

@btime mul!($v, $T, $u)

println("#===============================#")
println("3D Tensor Products")
println("#===============================#")

println("⊗(⊗(A, B), C)")

u = rand(12^3, 100)
v = rand(12^3, 100)

T = ⊗(⊗(A, B), C)
T = cache_operator(T, u)

mul!(v, T, u) # dunny
@btime mul!($v, $T, $u); #

println("⊗(A, ⊗(B, C))")

u = rand(12^3, 100)
v = rand(12^3, 100)

T = ⊗(A, ⊗(B, C))
T = cache_operator(T, u)

mul!(v, T, u) # dunny
@btime mul!($v, $T, $u); #

println("#===============================#")
nothing
