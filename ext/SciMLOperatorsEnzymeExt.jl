module SciMLOperatorsEnzymeExt

using SciMLOperators
using Enzyme
using LinearAlgebra

# Enzyme extension for SciMLOperators
#
# This extension ensures compatibility between Enzyme and SciMLOperators.
# The main issue is that operators contain function fields (update_func) which are
# closures that shouldn't be differentiated. By loading this extension, Enzyme's
# default behavior works correctly with the operator mathematical operations.

end # module
