using SciMLOperators, LinearAlgebra, Test, Zygote

using SciMLOperators: concretize

const ad_n = 3
const ad_u = [0.3, -0.2, 0.7]
const ad_v = [1.0, -2.0, 0.5]
const ad_t = 0.4
const ad_pmat = [
    0.0 2.0 -1.0
    1.0 0.0 0.5
    -0.25 0.75 0.0
]

ad_scalar() = ScalarOperator(0.0, (_, _, p, _) -> p)
ad_matrix() = MatrixOperator(ad_pmat)
ad_added_operator() = MatrixOperator(Matrix{Float64}(I, ad_n, ad_n)) + ad_scalar() * ad_matrix()

function ad_expected_scaled(p)
    return p .* (ad_pmat * ad_v)
end

function ad_expected_added(p)
    return (Matrix{Float64}(I, ad_n, ad_n) + p .* ad_pmat) * ad_v
end

@testset "AD semantic equivalence" begin
    p = 1.7

    @testset "ScalarOperator * MatrixOperator" begin
        L = ad_scalar() * ad_matrix()

        concretized_loss(p) = sum(concretize(update_coefficients(L, ad_u, p, ad_t)) * ad_v)
        direct_loss(p) = sum(L(ad_v, ad_u, p, ad_t))

        @test concretize(update_coefficients(L, ad_u, p, ad_t)) ≈ p .* ad_pmat
        @test L(ad_v, ad_u, p, ad_t) ≈ ad_expected_scaled(p)

        w = similar(ad_v)
        L(w, ad_v, ad_u, p, ad_t)
        @test w ≈ ad_expected_scaled(p)

        expected_grad = sum(ad_pmat * ad_v)
        @test only(Zygote.gradient(concretized_loss, p)) ≈ expected_grad
        @test only(Zygote.gradient(direct_loss, p)) ≈ expected_grad

        updated_L = update_coefficients(L, ad_u, p, ad_t)
        @test updated_L(ad_v, ad_u, p + 1, ad_t) ≈ ad_expected_scaled(p + 1)
        @test_throws ArgumentError update_coefficients!(updated_L, ad_u, p + 1, ad_t)
        updated_L(w, ad_v, ad_u, p + 1, ad_t)
        @test w ≈ ad_expected_scaled(p + 1)

        w .= 0.25
        updated_L(w, ad_v, ad_u, p + 1, ad_t, 2.0, 0.5)
        @test w ≈ 2 .* ad_expected_scaled(p + 1) .+ 0.125
    end

    @testset "MatrixOperator + ScalarOperator * MatrixOperator" begin
        L = ad_added_operator()

        concretized_loss(p) = sum(concretize(update_coefficients(L, ad_u, p, ad_t)) * ad_v)
        direct_loss(p) = sum(L(ad_v, ad_u, p, ad_t))

        @test concretize(update_coefficients(L, ad_u, p, ad_t)) ≈
            Matrix{Float64}(I, ad_n, ad_n) + p .* ad_pmat
        @test L(ad_v, ad_u, p, ad_t) ≈ ad_expected_added(p)

        w = similar(ad_v)
        L(w, ad_v, ad_u, p, ad_t)
        @test w ≈ ad_expected_added(p)

        expected_grad = sum(ad_pmat * ad_v)
        @test only(Zygote.gradient(concretized_loss, p)) ≈ expected_grad
        @test only(Zygote.gradient(direct_loss, p)) ≈ expected_grad
    end
end
