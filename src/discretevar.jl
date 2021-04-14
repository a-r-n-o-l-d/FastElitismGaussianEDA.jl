# overload model function

mutable struct DiscreteVariable{T} <: AbstractVariable
    wdth::T         # gap width
    vpop::Vector{T} # current values of the variable over population
    bstv::T         # best value so far
    modl            # probability model
    lini            # initial lower bound
end

function DiscreteVariable(r)
    T = eltype(r)
    l = first(r)
    u = last(r)
    wdth = step(r)
    vpop = Vector{T}()
    bstv = initv(T)
    modl = Uniform(0, (u - l) / wdth)
    DiscreteVariable(wdth, vpop, bstv, modl, l)
end

fit_model(v::DiscreteVariable, s) = fit_mle(Normal, v.vpop[s] ./ v.wdth)

function Base.rand(v::DiscreteVariable)
    if isa(v.modl, Uniform)
        s = (v.modl |> rand |> round) * v.wdth + v.lini
    else
        s = (v.modl |> rand |> round) * v.wdth
    end
    convert(eltype(v.vpop), s)
end

mutable struct DiscreteBoundedVariable{T} <: AbstractVariable
    lbnd::T         # lower bound
    ubnd::T         # upper bound
    wdth::T         # gap width
    vpop::Vector{T} # current values of the variable over population
    bstv::T         # best value so far
    modl            # probability model
end

function DiscreteBoundedVariable(r)
    T = eltype(r)
    lbnd = first(r)
    ubnd = last(r)
    wdth = step(r)
    vpop = Vector{T}()
    bstv = initv(T)
    modl = Uniform(0, (ubnd - lbnd) / wdth)
    DiscreteBoundedVariable(lbnd, ubnd, wdth, vpop, bstv, modl)
end

fit_model(v::DiscreteBoundedVariable, s) =
    truncated(fit_mle(Normal, (v.vpop[s] .- v.lbnd) ./ v.wdth), 0, (v.ubnd - v.lbnd) / v.wdth)

function Base.rand(v::DiscreteBoundedVariable)
    s = (v.modl |> rand |> round) * v.wdth + v.lbnd
    convert(eltype(v.vpop), s)
end