# Operator Overview

The operators in SciMLOperators.jl are instantiations of the `AbstractSciMLOperator`
interface. This is documented in [SciMLBase](https://diffeq.sciml.ai/stable/features/diffeq_operator/). Thus each of the operators
have the functions and traits as defined for the operator interface. 

## Operator Compositions

Multiplying two DiffEqOperators will build a `DiffEqOperatorComposition`, while
adding two DiffEqOperators will build a `DiffEqOperatorCombination`. Multiplying
a DiffEqOperator by a scalar will produce a `DiffEqScaledOperator`. All
will inherit the appropriate action.

### Efficiency of Composed Operator Actions

Composed operator actions utilize NNLib.jl in order to do cache-efficient
convolution operations in higher-dimensional combinations.

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
