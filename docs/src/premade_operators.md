# [Premade SciMLOperators](@id premade_operators)

## Direct Operator Definitions

```@docs
SciMLOperators.IdentityOperator
SciMLOperators.NullOperator
ScalarOperator
MatrixOperator
DiagonalOperator
SciMLOperators.BatchedDiagonalOperator
AffineOperator
AddVector
FunctionOperator
BlockDiagonalOperator
TensorProductOperator
SciMLOperators.:⊗
Base.kron
TensorSumOperator
kronsum
WOperator
SciMLOperators.StaticWOperator
```

## Lazy Scalar Operator Combination

```@docs
SciMLOperators.AddedScalarOperator
SciMLOperators.ComposedScalarOperator
SciMLOperators.InvertedScalarOperator
```

## Lazy Operator Combination

```@docs
SciMLOperators.ScaledOperator
SciMLOperators.AddedOperator
SciMLOperators.ComposedOperator
SciMLOperators.InvertedOperator
SciMLOperators.InvertibleOperator
SciMLOperators.AdjointOperator
SciMLOperators.TransposedOperator
```
