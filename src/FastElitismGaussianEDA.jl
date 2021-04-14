#=
To do :
- structure avec collection de variable
- add a termination criteria : MaxGens, MaxEvals, MaxEval (fitness threshold), MinStdDev
- add VectorVariable with multivariate normal (MvNormal in Distributions)
- add Categorical, BinaryVariable
- documentation
- examples : jupyter notebooks or doc(?)
=#

module FastElitismGaussianEDA

using Distributions


export Population, optimise!, optimum, state, neval
export VariableSet, names #, model
export ContinuousVariable, ContinuousBoundedVariable
export DiscreteVariable, DiscreteBoundedVariable

abstract type AbstractVariable end

include("variables.jl")
include("population.jl")

end # module
