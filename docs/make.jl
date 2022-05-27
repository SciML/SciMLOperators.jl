using Documenter, SciMLOperators

makedocs(
    sitename="SciMLOperators.jl",
    authors="Chris Rackauckas, Alex Jones",
    modules=[SciMLOperators],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://scimlbase.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Operators" => Any[
            "operators/matrix_free_operators.md",
        ]
    ]
)

deploydocs(
   repo = "github.com/SciML/SciMLOperators.jl.git";
   push_preview = true
)
