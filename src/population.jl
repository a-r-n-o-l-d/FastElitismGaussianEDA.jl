mutable struct Population
    ngen # number of generations
    psze # population size
    nslc # number of selected solutions in population
    nelt # number of elites to keep for the next generation
    ffun # fitness function
    fscl # fitness scale, set to -1 for minimisation
    vars # variables to pass to fitness function
    fpop # vector of fitness over population
    bstf # best fitness so far
    nevl # number of evaluation
end

function Population(ngen, psze, nslc, nelt, ffun, fscl, vars)
    psze < nslc && throw(ArgumentError(
        "Population size `psze` must be greater than number of selected individuals `nslc`."))
    nslc < nelt && throw(ArgumentError(
        "Number of selected individuals `nslc` must be greater than number of elites `nelt`."))
    fpop = Vector{Float64}()
    bstf = -Inf
    Population(ngen, psze, nslc, nelt, ffun, fscl, vars, fpop, bstf, 0)
end

Population(f, v::AbstractVariable; ngen, psze, nslc, nelt, fscl = 1) =
    Population(ngen, psze, nslc, nelt, f, fscl, VariableSet(v))

Population(f, v::Tuple; ngen, psze, nslc, nelt, fscl = 1) =
    Population(ngen, psze, nslc, nelt, f, fscl, VariableSet(v...))

Population(f, v::VariableSet; ngen, psze, nslc, nelt, fscl = 1) =
    Population(ngen, psze, nslc, nelt, f, fscl, v)

function update_population(p)
    i = sortperm(p.fpop, rev = true)
    b = i[1]
    s = i[1:p.nslc]
    e = i[1:p.nelt]
    for v ∈ p.vars
        update_variable!(v, b, s, e)
    end
    if p.fpop[b] > p.bstf
        p.bstf = p.fpop[b]
    end
    p.fpop = p.fpop[e]
    p
end

# add option: cb before or after update!(o)
# 1 seule callback
function optimise!(p; cbbs = (p) -> (), cbas = (p) -> (), mt = false)
    isempty(p.fpop) || throw(ArgumentError("A population can not be optimised more than once."))
    for _ ∈ 1:p.ngen
        psze = isempty(p.fpop) ? p.psze : p.psze - p.nelt
        evls = []
        if mt
            for _ ∈ 1:Threads.nthreads()
                push!(evls, [])
            end
            Threads.@threads for _ ∈ 1:psze
                x = [rand(v) for v ∈ p.vars]
                f = p.fscl * p.ffun(x...)
                push!(evls[Threads.threadid()], (x, f))
            end
            evls = vcat(evls...)
        else
            for _ ∈ 1:psze
                x = [rand(v) for v ∈ p.vars]
                f = p.fscl * p.ffun(x...)
                push!(evls, (x, f))
            end
        end
        for e ∈ evls
            x, f = e
            for (v, s) ∈ zip(p.vars, x)
                push!(v.vpop, s)
            end
            push!(p.fpop, f)
            p.nevl += 1
        end
        cbbs(p)
        update_population(p)
        cbas(p)
        # if termination criteria break
    end
    p
end

function optimum(p::Population)
    if length(p.vars) == 1
        return (p.vars.bstv, p.bstf / p.fscl)
    end
    nm = names(p.vars)
    bv = tuple([v.bstv for v ∈ p.vars]...)
    (NamedTuple{nm}(bv), p.bstf / p.fscl)
end

# models

# current state of population : fitness values and variable values (summary)
# (dictionnary : fitness => fpop, name = var)
# :fitness : mean(fpop ./ fscl) std(fpop ./ fscl)
# :variables : :mean :std :model
# model : DiscreteVariable v.modl * v.wdth (arithmetic on didtributions are allowed)
#         DiscreteBoundedVariable v.modl * v.wdth + v.lbnd
function state(p)
    model(v::AbstractVariable) = v.modl
    model(v::DiscreteVariable) = v.modl * v.wdth
    model(v::DiscreteBoundedVariable) = v.modl * v.wdth + v.lbnd
    nm = []
    st = []
    for (n, v) ∈ zip(names(p.vars), p.vars)
        push!(nm, n)
        #push!(st, v.vpop)
        m = model(v)
        push!(st, (mean = mean(m), std = std(m), model = m))
    end
    push!(nm, :fitness)
    fpop = p.fpop ./ p.fscl
    push!(st, (mean = mean(fpop), std = std(fpop)))
    NamedTuple{tuple(nm...)}(tuple(st...))
end

neval(p) = p.nevl