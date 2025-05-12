using SafeTestsets

@time begin
    @time @safetestset "Scalar Operators" begin
        include("scalar/scalar_new.jl")
        include("scalar/scalar_modified.jl")
    end
    @time @safetestset "Basic Operators" begin
        include("basic/basic_new.jl")
        # basic modified checks for allocations which we need to optimize
        # include("basic/basic_modified.jl") 
    end
    @time @safetestset "Matrix Operators" begin
        include("matrix/matrix_new.jl")
        include("matrix/matrix_modified.jl")
    end
    @time @safetestset "Function Operator" begin
        include("func/func_new.jl")
        include("func/func_modified.jl")
    end
    @time @safetestset "Full tests" begin
        include("total/total_new.jl")
    end
    @time @safetestset "Zygote.jl" begin
        #  Zygote test will be focused upon later.
        # include("zygote/zygote.jl")
    end
end
