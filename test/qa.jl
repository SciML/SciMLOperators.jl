using SciMLOperators, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(SciMLOperators)
    Aqua.test_ambiguities(SciMLOperators, recursive = false)
    Aqua.test_deps_compat(SciMLOperators)
    Aqua.test_piracies(SciMLOperators,
        treat_as_own = [])
    Aqua.test_project_extras(SciMLOperators)
    Aqua.test_stale_deps(SciMLOperators)
    Aqua.test_unbound_args(SciMLOperators)
    Aqua.test_undefined_exports(SciMLOperators)
end
