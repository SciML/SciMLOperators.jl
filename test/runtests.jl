using SafeTestsets

@time begin
    @time @safetestset "Quality Assurance" begin
        include("qa.jl")
    end
    @time @safetestset "Scalar Operators" begin
        include("scalar.jl")
    end
    @time @safetestset "Basic Operators" begin
        include("basic.jl")
    end
    @time @safetestset "Matrix Operators" begin
        include("matrix.jl")
    end
    @time @safetestset "Function Operator" begin
        include("func.jl")
    end
    @time @safetestset "Full tests" begin
        include("total.jl")
    end
    @time @safetestset "Zygote.jl" begin
        include("zygote.jl")
    end
end
