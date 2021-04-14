struct VariableSet{T <: AbstractVariable}
    vars::Vector{T}
    nmes
end

function VariableSet(vars::AbstractVariable...)
    nmes = Dict([Symbol(i) => i for i ∈ eachindex(vars)])
    VariableSet([vars...], nmes)
end

function VariableSet(vars::Pair{Symbol,}...)
    nmes = Dict([first(vars[i]) => i for i ∈ eachindex(vars)])
    vars = [last(vars[i]) for i ∈ eachindex(vars)]
    VariableSet(vars, nmes)
end

function Base.names(vs)
    nm = collect(keys(vs.nmes))
    idx = collect(values(vs.nmes))
    nn = copy(nm)
    for i ∈ eachindex(idx)
        nn[idx[i]] = nm[i]
    end
    tuple(nn...)
end

Base.getindex(vs::VariableSet, i::Int) = vs.vars[i]

Base.getindex(vs::VariableSet, n) = vs.vars[vs.nmes[n]]

Base.setindex!(vs::VariableSet, w, i::Int) = (vs.vars[i] = w)

Base.setindex!(vs::VariableSet, w, n) = (vs.vars[vs.nmes[n]] = w)

Base.length(vs::VariableSet) = length(vs.vars)

Base.firstindex(vs::VariableSet) = 1

Base.lastindex(vs::VariableSet) = length(vs)

Base.iterate(vs::VariableSet) = iterate(vs, 1)

function Base.iterate(vs::VariableSet, st)
    st > length(vs) && return nothing
    (vs[st], st + 1)
end

function Base.push!(vs::VariableSet, var::Pair)
    push!(vs.vars, last(var))
    n = length(vs)
    push!(vs.nmes, first(var) => n)
    vs
end

function Base.push!(vs::VariableSet, var::AbstractVariable)
    push!(vs.vars, var)
    n = length(vs)
    push!(vs.nmes, Symbol(n) => n)
    vs
end

function Base.push!(vs::VariableSet, vars::AbstractVariable...)
    for v ∈ vars
        push!(vs.vars, v)
    end
    vs
end

