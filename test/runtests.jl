using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")
const is_TRAVIS = haskey(ENV, "TRAVIS")

@time begin
    if GROUP == "All" || GROUP == "OperatorInterface"
        @time @safetestset "Basic Operators" begin
            include("basic.jl")
        end
        @time @safetestset "DiffEq Operators" begin
            include("sciml.jl")
        end
#       @time @safetestset "Matrix-Free Operators" begin
#           include("matrixfree.jl")
#       end
#       @time @safetestset "Composite Operators Interface" begin
#           include("composite_operators_interface.jl")
#       end
    end
end
