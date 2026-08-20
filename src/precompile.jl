@setup_workload begin
    v = [1.0, 2.0]
    matrix_operator = MatrixOperator([2.0 1.0; 1.0 3.0])
    diagonal_operator = DiagonalOperator([2.0, 3.0])
    identity_operator = IdentityOperator(2)
    scalar_operator = ScalarOperator(2.0)
    function_operator = FunctionOperator(
        (v, u, p, t) -> 2 .* v,
        v;
        ifcache = false,
        isconstant = true,
        islinear = true
    )
    added_operator = matrix_operator + diagonal_operator
    composed_operator = matrix_operator * diagonal_operator
    block_operator = BlockDiagonalOperator(matrix_operator, diagonal_operator)
    tensor_operator = kron(identity_operator, diagonal_operator)

    @compile_workload begin
        matrix_operator * v
        mul!(similar(v), matrix_operator, v)
        diagonal_operator(v, nothing, nothing, 0.0)
        scalar_operator * v
        function_operator * v
        added_operator * v
        composed_operator * v
        (matrix_operator ∘ diagonal_operator) * v
        cache_operator(composed_operator, v) * v
        block_operator * [1.0, 2.0, 3.0, 4.0]
        tensor_operator * [1.0, 2.0, 3.0, 4.0]
    end
end
