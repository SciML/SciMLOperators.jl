using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: InvertibleOperator, ⊗
using FFTW

Random.seed!(0)
N = 8
K = 19

@testset "MatrixOperator, InvertibleOperator" begin
    u = rand(N,K)
    p = nothing
    t = 0
    α = rand()
    β = rand()

    A  = rand(N,N)
    At = A'

    AA  = MatrixOperator(A)
    AAt = AA'

    @test AA  isa MatrixOperator
    @test AAt isa MatrixOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa InvertibleOperator
    @test FFt isa InvertibleOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(MatrixOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)
    @test At * u ≈ AAt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u

    v=rand(N,K); @test mul!(v, AA, u) ≈ A * u
    v=rand(N,K); w=copy(v); @test mul!(v, AA, u, α, β) ≈ α*A*u + β*w
end

@testset "MatrixOperator update test" begin
    u = rand(N,K)
    p = rand(N)
    t = rand()

    L = MatrixOperator(zeros(N,N);
                       update_func= (A,u,p,t) -> (A .= p*p'; nothing)
                      )

    A = p*p'
    ans = A * u
    @test L(u,p,t) ≈ ans
    v=copy(u); @test L(v,u,p,t) ≈ ans
end

@testset "DiagonalOperator update test" begin
    u = rand(N,K)
    p = rand(N)
    t = rand()

    D = DiagonalOperator(zeros(N);
                         update_func= (diag,u,p,t) -> (diag .= p*t; nothing)
                        )

    ans = Diagonal(p*t) * u
    @test D(u,p,t) ≈ ans
    v=copy(u); @test D(v,u,p,t) ≈ ans
end

@testset "Batched Diagonal Operator" begin
    u = rand(N,K)
    d = rand(N,K)
    α = rand()
    β = rand()

    L = DiagonalOperator(d)

    @test L * u ≈ d .* u
    v=rand(N,K); @test mul!(v, L, u) ≈ d .* u
    v=rand(N,K); w=copy(v); @test mul!(v, L, u, α, β) ≈ α*(d .* u) + β*w

    @test L \ u ≈ d .\ u
    v=rand(N,K); @test ldiv!(v, L, u) ≈ d .\ u
    v=copy(u); @test ldiv!(L, u) ≈ d .\ v
end

@testset "AffineOperator" begin
    u = rand(N,K)
    A = rand(N,N)
    B = rand(N,N)
    D = Diagonal(A)
    b = rand(N,K)
    α = rand()
    β = rand()

    L = AffineOperator(MatrixOperator(A), MatrixOperator(B), b)

    @test L * u ≈ A * u + B*b
    v=rand(N,K); @test mul!(v, L, u) ≈ A*u + B*b
    v=rand(N,K); w=copy(v); @test mul!(v, L, u, α, β) ≈ α*(A*u + B*b) + β*w

    L = AffineOperator(MatrixOperator(D), MatrixOperator(B), b)
    @test L \ u ≈ D \ (u - B * b)
    v=rand(N,K); @test ldiv!(v, L, u) ≈ D \ (u-B*b)
    v=copy(u); @test ldiv!(L, u) ≈ D \ (v-B*b)

    L = AddVector(b)
    @test L * u ≈ u + b
    @test L \ u ≈ u - b
    v=rand(N,K); @test mul!(v, L, u) ≈  u + b
    v=rand(N,K); w=copy(v); @test mul!(v, L, u, α, β) ≈ α*(u + b) + β*w
    v=rand(N,K); @test ldiv!(v, L, u) ≈  u - b
    v=copy(u); @test ldiv!(L, u) ≈  v - b

    L = AddVector(MatrixOperator(B), b)
    @test L * u ≈ u + B * b
    @test L \ u ≈ u - B * b
    v=rand(N,K); @test mul!(v, L, u) ≈  u + B * b
    v=rand(N,K); w=copy(v); @test mul!(v, L, u, α, β) ≈ α*(u + B * b) + β*w
    v=rand(N,K); @test ldiv!(v, L, u) ≈  u - B * b
    v=copy(u); @test ldiv!(L, u) ≈  v - B * b
end

@testset "AffineOperator update test" begin
    A = rand(N,N)
    B = rand(N,N)
    b = rand(N,K)
    u = rand(N,K)
    p = rand(N)
    t = rand()

    L = AffineOperator(A, B, b;
                       update_func= (b,u,p,t) -> (b .= Diagonal(p*t)*b; nothing)
                      )

    b = Diagonal(p*t)*b
    ans = A * u + B * b
    @test L(u,p,t) ≈ ans
    b = Diagonal(p*t)*b
    ans = A * u + B * b
    v=copy(u); @test L(v,u,p,t) ≈ ans
end

@testset "TensorProductOperator" begin
for square in [false, true] #for K in [1, K]
    m1, n1 = 3 , 5
    m2, n2 = 7 , 11
    m3, n3 = 13, 17

    if square
        n1, n2, n3 = m1, m2, m3
    end

    A = rand(m1, n1)
    B = rand(m2, n2)
    C = rand(m3, n3)
    α = rand()
    β = rand()

    AB  = kron(A, B)
    ABC = kron(A, B, C)

    # Inputs
    u2 = rand(n1*n2, K)
    u3 = rand(n1*n2*n3, K)
    # Outputs
    v2 = rand(m1*m2, K)
    v3 = rand(m1*m2*m3, K)

    # Outputs
    v2 = rand(m1*m2, K)
    v3 = rand(m1*m2*m3, K)

    opAB  = TensorProductOperator(A, B)
    opABC = TensorProductOperator(A, B, C)

    @test opAB  isa TensorProductOperator
    @test opABC isa TensorProductOperator

    @test AB ≈ convert(AbstractMatrix, opAB)
    @test ABC ≈ convert(AbstractMatrix, opABC)

    # factorization tests
    opAB_F = factorize(opAB)
    opABC_F = factorize(opABC)

    @test opAB_F isa TensorProductOperator
    @test opABC_F isa TensorProductOperator

    @test AB ≈ convert(AbstractMatrix, opAB_F)
    @test ABC ≈ convert(AbstractMatrix, opABC_F)

    @test AB  * u2 ≈ opAB  * u2
    @test ABC * u3 ≈ opABC * u3

    @test AB  \ v2 ≈ opAB  \ v2 ≈ opAB_F  \ v2
    @test ABC \ v3 ≈ opABC \ v3 ≈ opABC_F \ v3

    opAB  = cache_operator(opAB,  u2)
    opABC = cache_operator(opABC, u3)

    opAB_F  = cache_operator(opAB_F,  u2)
    opABC_F = cache_operator(opABC_F, u3)

    N2 = n1*n2
    N3 = n1*n2*n3
    M2 = m1*m2
    M3 = m1*m2*m3

    v2=rand(M2,K); @test mul!(v2, opAB , u2) ≈ AB  * u2
    v3=rand(M3,K); @test mul!(v3, opABC, u3) ≈ ABC * u3

    v2=rand(M2,K); w2=copy(v2); @test mul!(v2, opAB , u2, α, β) ≈ α*AB *u2 + β*w2
    v3=rand(M3,K); w3=copy(v3); @test mul!(v3, opABC, u3, α, β) ≈ α*ABC*u3 + β*w3

    if square
        u2=rand(N2,K); @test ldiv!(u2, opAB_F , v2) ≈ AB  \ v2
        u3=rand(N3,K); @test ldiv!(u3, opABC_F, v3) ≈ ABC \ v3

        v2=copy(u2); @test ldiv!(opAB_F , u2) ≈ AB  \ v2
        v3=copy(u3); @test ldiv!(opABC_F, u3) ≈ ABC \ v3
    else # TODO
        u2=rand(N2,K); 
        if VERSION < v"1.9-"
            @test_broken ldiv!(u2, opAB_F , v2) ≈ AB  \ v2
        else
            @test ldiv!(u2, opAB_F , v2) ≈ AB  \ v2 # fails
        end
        u3=rand(N3,K); @test_broken ldiv!(u3, opABC_F, v3) ≈ ABC \ v3 # errors
    end

end #end
end
#
