# SciMLOperators.jl: The SciML Operators Interface

Many functions, from linear solvers to differential equations, require the use of
matrix-free operators in order to achieve maximum performance in many scenarios.
SciMLOperators.jl defines the abstract interface for how operators in the SciML
ecosystem are supposed to be defined. It gives the common set of functions and
traits which solvers can rely on for properly performing their tasks. Along with
that, SciMLOperators.jl provides definitions for the basic standard operators
which are used in building blocks for most tasks, both simplifying the use of operators
while also demonstrating to users how such operators can be built and used in practice.

## Why SciMLOperators?

SciMLOperators.jl has the design that is required in order to be used in all scenarios
of equation solvers. For example, Magnus integrators for differential equations
require defining an operator ``u' = A(t)u``, while Munthe-Kaas methods require defining
operators of the form ``u' = A(u)u``. Thus the operators need some form of time and
state dependence which the solvers can update and query when they are non-constant
(`update_coefficients!`). Additionally, the operators need the ability to act like
"normal" functions for equation solvers. For example, if `A(u,p,t)` has the same
operation as `update_coefficients(A,u,p,t); A*u`, then `A` can be used in any place where
a differential equation definition `f(u,p,t)` is used without requring the user or solver
to do any extra work. 

Thus while previous good efforts for matrix-free operators have existed in the Julia ecosystem, 
such as [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), those operator
interfaces lack these aspects in order to actually be fully seamless with downstream equation
solvers. This necessitates the definition and use of an extended operator interface with all
of these properties, hence the `AbstractSciMLOperator` interface.

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - On the Julia Discourse forums (look for the [modelingtoolkit tag](https://discourse.julialang.org/tag/modelingtoolkit)
    - See also [SciML Community page](https://sciml.ai/community/)
