using SciMLOperators, Aqua, JET, Test

@testset "Aqua" begin
    Aqua.test_all(SciMLOperators)
end

@testset "JET" begin
    JET.test_package(SciMLOperators; target_defined_modules = true)
end
