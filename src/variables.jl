include("varset.jl")
include("continuousvar.jl")
include("discretevar.jl")

initv(T) = typemin(T)

#=
function initv(T)
    T == Float16 && return NaN16
    T == Float32 && return NaN32
    T == Float64 && return NaN
    typemin(T)
end
=#

Base.values(v::AbstractVariable) = v.vpop

#model(v) = v.modl

function update_variable!(v, b, s, e)
    # Store best value
    v.bstv = v.vpop[b]
    # Fit model with the best selected individuals (solutions)
    v.modl = fit_model(v, s)
    # Retain the elite individuals for the next generation
    v.vpop = v.vpop[e]
    v
end

Base.rand(v::AbstractVariable) = convert(eltype(v.vpop), v.modl |> rand)