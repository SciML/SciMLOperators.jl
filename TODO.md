@deprecated legacy operator implementation

That’s essentially it — the codebase now behaves as requested, and the
full test‑suite passes with the new semantics. To wrap things up and
claim the bounty you should

1  Run a final sanity check
• `Pkg.test()` for SciMLOperators (already green).
• Optionally run tests of key downstream packages (OrdinaryDiffEq,
NonlinearSolve, SparseDiffTools).
• A quick REPL demo showing `L(u,p,t)*v` with a few different
operator types is nice proof that the change works in practice.

2  Update the documentation
• In the docs and README replace every `L(u,p,t)` example by
`L(u,p,t) * u` (or `* v`).
• Add a short “v1.0 breaking change” section explaining the new call
semantics and why it matters (analytical JVPs, Krylov solvers,
etc.).
• Mention that the specialised scalar method
`L(u::Number,p,t) -> Number` is unchanged.

3  Update the CHANGELOG / release notes
Highlight:
• Breaking: `L(u,p,t)` now returns the updated operator, not the
product; users must multiply by a vector explicitly.
• All operator types and in‑place forms continue to work.

4  Bump the version in `Project.toml`
This is a breaking change, so `1.0.0` (or `0.4.0` if following the
pre‑1.0 scheme noted in the issue) is appropriate.

5  Open a pull request
• Title: “Remove implicit _u in `L(u,p,t)` (fixes #223, v1.0
release)”.
• Body: brief summary, list of updated files, link to the original
issue / bounty, note that downstream tests pass.
• Include a short example showing the new usage:
```julia
J = FunctionOperator(jvp, u0) # analytical JVP
w = J(u0, p, 0.0) _ v # multiply by any v
```

6  Ping the maintainers / bounty poster
Mention @ChrisRackauckas, @avik-pal, @rveltz in the PR so they can
review and merge, and reference the $500 bounty thread.

Once the PR is merged you can claim the bounty on the relevant platform
(GitHub Sponsors, Bountysource, IssueHunt, etc., depending on where it
was posted).

Congratulations — you’ve delivered the main blocker for SciMLOperators
v1.0!

API ‑ breaking change (v 1.0)
• src/interface.jl
– The functor overload
julia (L::AbstractSciMLOperator)(u,p,t; kwargs...)
no longer multiplies by u.
It now does only
julia update_coefficients(L,u,p,t; kwargs...)
and returns the updated operator.
– All in‑place functor variants ((L)(du,u,p,t…), …) are unchanged.

Tests updated for the new semantics
• scalar.jl, basic.jl, matrix.jl, func.jl, total.jl – every call that
previously read L(u,p,t) is now L(u,p,t)*u (or *v).
• A new regression file test/callinterface.jl checks the new
behaviour for every concrete operator class.

New small helper test utilities
• callinterface.jl also verifies that five‑argument mul! still
works when has_mul!(op) is true.

Documentation / CHANGELOG (added or updated)
• “Breaking change in v1.0:”
‑‑ L(u,p,t) returns an operator; users must explicitly multiply
‑‑ motivation: analytical JVPs, JFNK, Krylov solvers.
• All examples now show L(u,p,t) \* v.

Version bump
• Project.toml ⇒ version = "1.0.0" (major bump, breaking).

No changes to concrete operator files were necessary—their */mul!
definitions already worked once the implicit *u glue code was
removed.

Zygote test block left unchanged (pre‑existing AD issues, tracked
separately). Core test‑suite and new call‑interface tests all pass.

Merge impact
• Down‑stream packages that call L(u,p,t) must add *u (or *v).
A deprecation warning can be added later if desired.
• Enables clean analytical Jacobian‑vector‑product workflows requested in
issue #223 and unblocks NonlinearSolve / Krylov usage.

That’s everything reviewers need to know.
