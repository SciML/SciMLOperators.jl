using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")
const is_TRAVIS = haskey(ENV, "TRAVIS")

@time begin
if GROUP == "All" || GROUP == "OperatorInterface"
    @time @safetestset "Scalar Operators" begin include("scalar.jl") end
    @time @safetestset "Basic Operators" begin include("basic.jl") end
    @time @safetestset "Matrix Operators" begin include("matrix.jl") end
    @time @safetestset "Function Operator" begin include("func.jl") end
    @time @safetestset "Full tests" begin include("total.jl") end
end
end
