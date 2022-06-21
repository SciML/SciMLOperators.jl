using SciMLOperators, LinearAlgebra, BenchmarkTools

println("#===============================#")
println("TensorProduct(A, B)")
println("#===============================#")

u = rand(12^2, 100)
v = rand(12^2, 100)

A = rand(12, 12) # outer
B = rand(12, 12) # inner

T = TensorProductOperator(A, B)
T = cache_operator(T, u)

@btime mul!($v, $T, $u); #

#println("#===============================#")
#println("TensorProduct(A, B, C)")
#println("#===============================#")
#
#A = TensorProductOperator(rand(12,12), rand(12,12), rand(12,12))
#
#u = rand(12^3, 100)
#v = rand(12^3, 100)
#
#A = cache_operator(A, u)
#
#mul!(v, A, u) # dunny
#@btime mul!($v, $A, $u); #

println("#===============================#")
println("TensorProduct(TensorProduct(A, B), C)")
println("#===============================#")

A = TensorProductOperator(TensorProductOperator(rand(12,12), rand(12,12)), rand(12,12))

u = rand(12^3, 100)
v = rand(12^3, 100)

A = cache_operator(A, u)

mul!(v, A, u) # dunny
@btime mul!($v, $A, $u); #

println("#===============================#")
println("TensorProduct(A, TensorProduct(B, C))")
println("#===============================#")

A = TensorProductOperator(rand(12,12), TensorProductOperator(rand(12,12), rand(12,12)))

u = rand(12^3, 100)
v = rand(12^3, 100)

A = cache_operator(A, u)

mul!(v, A, u) # dunny
@btime mul!($v, $A, $u); #

println("#===============================#")
nothing
