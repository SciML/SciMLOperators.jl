using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")
const is_TRAVIS = haskey(ENV, "TRAVIS")

@time begin
    if GROUP == "All" || GROUP == "OperatorInterface"
        @time @safetestset "Basic Operators" begin include("basic.jl") end
        @time @safetestset "SciML Operators" begin include("sciml.jl") end
        @time @safetestset "Function Operator" begin include("func.jl") end
    end
end
