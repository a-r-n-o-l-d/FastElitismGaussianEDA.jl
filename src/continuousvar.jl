mutable struct ContinuousVariable{T <: Real} <: AbstractVariable
    vpop::Vector{T} # current values of the variable over population
    bstv::T         # best value so far
    modl            # probability model
end

function ContinuousVariable(l::T, u::T) where T <: AbstractFloat
    vpop = Vector{T}()
    bstv = initv(T)
    modl = Uniform{T}(l, u)
    ContinuousVariable(vpop, bstv, modl)
end

function ContinuousVariable(l::T, u::T) where T <: Real
    l = float(l)
    u = float(u)
    ContinuousVariable(l, u)
end

function ContinuousVariable(l::T1, u::T2) where {T1 <: Real, T2 <: Real}
    l, u = promote(l, u)
    ContinuousVariable(l, u)
end

fit_model(v::ContinuousVariable, s) = fit_mle(Normal, v.vpop[s])


mutable struct ContinuousBoundedVariable{T <: Real} <: AbstractVariable
    lbnd::T         # lower bound
    ubnd::T         # upper bound
    vpop::Vector{T} # current values of the variable over population
    bstv::T         # best value so far
    modl            # probability model
end

function ContinuousBoundedVariable(lbnd::T, ubnd::T) where T <: AbstractFloat
    vpop = Vector{T}()
    bstv = initv(T)
    modl = Uniform{T}(lbnd, ubnd)
    ContinuousBoundedVariable(lbnd, ubnd, vpop, bstv, modl)
end

function ContinuousBoundedVariable(lbnd::T, ubnd::T) where T <: Real
    lbnd = Float64(lbnd)
    ubnd = Float64(ubnd)
    ContinuousBoundedVariable(lbnd, ubnd)
end

function ContinuousBoundedVariable(lbnd::T1, ubnd::T2) where {T1 <: Real, T2 <: Real}
    lbnd, ubnd = promote(lbnd, ubnd)
    ContinuousBoundedVariable(lbnd, ubnd)
end

fit_model(v::ContinuousBoundedVariable, s) = truncated(fit_mle(Normal, v.vpop[s]), v.lbnd, v.ubnd)