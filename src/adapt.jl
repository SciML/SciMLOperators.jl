#
# Adapt.jl integration for SciMLOperators
#
# Provides Adapt.adapt_structure methods for operators that contain array storage
# or can transitively contain adapted operators.
#

# ============================================================================
# Direct Array-Backed Operators (store arrays that need adaptation)
# ============================================================================

function Adapt.adapt_structure(to, L::MatrixOperator)
    A_adapted = Adapt.adapt(to, L.A)
    return MatrixOperator(
        A_adapted,
        L.update_func,
        L.update_func!
    )
end

function Adapt.adapt_structure(to, L::BatchedDiagonalOperator)
    diag_adapted = Adapt.adapt(to, L.diag)
    return BatchedDiagonalOperator(
        diag_adapted,
        L.update_func,
        L.update_func!
    )
end

function Adapt.adapt_structure(to, L::AffineOperator)
    A_adapted = Adapt.adapt_structure(to, L.A)
    B_adapted = Adapt.adapt_structure(to, L.B)
    b_adapted = Adapt.adapt(to, L.b)
    return AffineOperator(
        A_adapted,
        B_adapted,
        b_adapted,
        L.update_func,
        L.update_func!
    )
end

function Adapt.adapt_structure(to, L::FunctionOperator)
    u_adapted = Adapt.adapt(to, L.u)
    p_adapted = Adapt.adapt(to, L.p)
    t_adapted = Adapt.adapt(to, L.t)
    cache_adapted = Adapt.adapt(to, L.cache)

    return set_cache(
        set_t(
            set_p(
                set_u(L, u_adapted),
                p_adapted
            ),
            t_adapted
        ),
        cache_adapted
    )
end

# ============================================================================
# Recursive Wrappers (can contain adapted operators or array caches)
# ============================================================================

function Adapt.adapt_structure(to, L::ScaledOperator)
    L_adapted = Adapt.adapt_structure(to, L.L)
    return ScaledOperator(L.λ, L_adapted)
end

function Adapt.adapt_structure(to, L::AddedOperator)
    return AddedOperator(Adapt.adapt_structure(to, L.ops))
end

function Adapt.adapt_structure(to, L::ComposedOperator)
    ops_adapted = Adapt.adapt_structure(to, L.ops)
    cache_adapted = Adapt.adapt(to, L.cache)
    return ComposedOperator(ops_adapted...; cache = cache_adapted)
end

function Adapt.adapt_structure(to, L::InvertedOperator)
    L_adapted = Adapt.adapt_structure(to, L.L)
    cache_adapted = Adapt.adapt(to, L.cache)
    return InvertedOperator(L_adapted; cache = cache_adapted)
end

function Adapt.adapt_structure(to, L::InvertibleOperator)
    L_adapted = Adapt.adapt_structure(to, L.L)
    F_adapted = Adapt.adapt_structure(to, L.F)
    return InvertibleOperator(L_adapted, F_adapted)
end

function Adapt.adapt_structure(to, L::BlockDiagonalOperator)
    return BlockDiagonalOperator(Adapt.adapt_structure(to, L.ops))
end

function Adapt.adapt_structure(to, L::AdjointOperator)
    L_adapted = Adapt.adapt_structure(to, L.L)
    return AdjointOperator(L_adapted)
end

function Adapt.adapt_structure(to, L::TransposedOperator)
    L_adapted = Adapt.adapt_structure(to, L.L)
    return TransposedOperator(L_adapted)
end

function Adapt.adapt_structure(to, L::TensorProductOperator)
    ops_adapted = Adapt.adapt_structure(to, L.ops)
    cache_adapted = Adapt.adapt(to, L.cache)
    return TensorProductOperator(ops_adapted...; cache = cache_adapted)
end

function Adapt.adapt_structure(to, L::TensorSumOperator)
    ops_adapted = Adapt.adapt_structure(to, L.ops)
    return TensorSumOperator(ops_adapted...)
end

# ============================================================================
# Scalar Operators (no array storage, handled by generic Adapt fallback)
# ============================================================================
# ScalarOperator, AddedScalarOperator, ComposedScalarOperator, InvertedScalarOperator
# do not store arrays, so they use Adapt's default struct adaptation which
# recursively adapts non-array fields and leaves scalars unchanged.

# ============================================================================
# Explicitly Skipped (no arrays, no need for custom adaptation)
# ============================================================================
# IdentityOperator, NullOperator: store only integers/dimensions
# These use Adapt's default struct adaptation (which is fine since no arrays).
#
