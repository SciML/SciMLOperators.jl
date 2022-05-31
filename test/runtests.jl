using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")
const is_TRAVIS = haskey(ENV, "TRAVIS")

@time begin
    if GROUP == "All" || GROUP == "OperatorInterface"
        @time @safetestset "Basic Operators" begin
            include("operators/basic_operators.jl")
        end
#       @time @safetestset "Matrix-Free Operators" begin
#           include("operators/matrixfree.jl")
#       end
#       @time @safetestset "Composite Operators Interface" begin
#           include("operators/composite_operators_interface.jl")
#       end
#       @time @safetestset "DiffEqOperator tests" begin
#           include("operators/diffeqoperator.jl")
#       end
    end
end
