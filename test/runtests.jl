#=
To do :
- verify support -Inf Inf for truncated variable bounds (marche pas avec uniform)
=#

using Distributions, Test
using FastElitismGaussianEDA
using FastElitismGaussianEDA: AbstractVariable, rand, fit_model, update_variable!

@testset "Variable constructors and type consistency" begin
    v = ContinuousVariable(1.0, 2.0)
    @test isa(rand(v), Float64)
    v = ContinuousVariable(1f0, 2f0)
    @test isa(rand(v), Float32)
    v = ContinuousVariable(1f0, 2.0)
    @test isa(rand(v), Float64)
    v = ContinuousVariable(1, 2)
    @test isa(rand(v), Float64)
    v = ContinuousVariable(Float16(1), Float16(2))
    @test isa(rand(v), Float16)

    v = ContinuousBoundedVariable(1.0, 2.0)
    @test isa(rand(v), Float64)
    v = ContinuousBoundedVariable(1f0, 2f0)
    @test isa(rand(v), Float32)
    v = ContinuousBoundedVariable(1f0, 2.0)
    @test isa(rand(v), Float64)
    v = ContinuousBoundedVariable(1, 2)
    @test isa(rand(v), Float64)
    v = ContinuousBoundedVariable(Float16(1), Float16(2))
    @test isa(rand(v), Float16)

    rng = 1.0:0.01:2.0
    v = DiscreteVariable(rng)
    @test isa(rand(v), eltype(rng))
    rng = 1f0:1f-1:2f0
    v = DiscreteVariable(rng)
    @test isa(rand(v), eltype(rng))
    rng = 1:1:5
    v = DiscreteVariable(rng)
    @test isa(rand(v), eltype(rng))

    rng = 1.0:0.01:2.0
    v = DiscreteBoundedVariable(rng)
    @test isa(rand(v), eltype(rng))
    rng = 1f0:1f-1:2f0
    v = DiscreteBoundedVariable(rng)
    @test isa(rand(v), eltype(rng))
    rng = 1:1:5
    v = DiscreteBoundedVariable(rng)
    @test isa(rand(v), eltype(rng))
end

@testset "VariableSet" begin
    v1 = ContinuousVariable(-10, 10)
    v2 = ContinuousBoundedVariable(-10, 10)
    v3 = DiscreteVariable(-10:0.1:10)
    v4 = DiscreteBoundedVariable(-10:0.1:10)

    s = VariableSet(v1, v2, v3, v4)
    @test isa(s.vars, AbstractArray)
    @test first(s) === v1
    @test last(s) === v4
    @test s[2] === v2
    tmp = DiscreteBoundedVariable(-10:0.1:10)
    s[3] = tmp
    @test s[3] === tmp
    @test length(values.(s)) == 4

    s = VariableSet(:x => ContinuousVariable(-10, 10), 
                    :y => DiscreteVariable(-10:0.1:10),
                    :z => DiscreteBoundedVariable(-10:0.1:10))
    @test isa(s[1], ContinuousVariable)
    @test isa(s[2], DiscreteVariable)
    @test isa(s[3], DiscreteBoundedVariable)
    @test isa(s[:x], ContinuousVariable)
    @test isa(s[:y], DiscreteVariable)
    @test isa(s[:z], DiscreteBoundedVariable)
    tmp = DiscreteBoundedVariable(-10:0.1:10)
    s[:z] = tmp
    @test s[3] === tmp
    for v ∈ s
        @test isa(v, AbstractVariable)
    end
    push!(s, ContinuousVariable(-10, 10))
    @test length(s) == 4
    push!(s, ContinuousVariable(-10, 10), ContinuousBoundedVariable(-10, 10))
    @test length(s) == 6
end

@testset "Variable sampling" begin
    v = ContinuousVariable(1.0, 2.0)
    @test 1.0 <= rand(v) <= 2.0

    v = ContinuousBoundedVariable(1.0, 2.0)
    @test 1.0 <= rand(v) <= 2.0

    v = DiscreteVariable(1.0:0.01:2.0)
    @test 1.0 <= rand(v) <= 2.0

    v = DiscreteBoundedVariable(1.0:0.01:2.0)
    @test 1.0 <= rand(v) <= 2.0
end

@testset "Population constructor and initialisation" begin
    f(x) = -x^2

    v = ContinuousVariable(-10, 10)
    p = Population(f, v, ngen = 100, psze = 30, nslc = 10, nelt = 3)
    @test length(p.vars) == 1

    g(x, y) = -x^2 - y^2
    v1 = ContinuousVariable(-10, 10)
    v2 = ContinuousBoundedVariable(-10, 10)
    p = Population(g, (v1, v2), ngen = 100, psze = 30, nslc = 10, nelt = 3)
    @test length(p.vars) == 2

    @test_throws ArgumentError Population(f, (v1, v2); ngen = 100, psze = 30, nslc = 100, nelt = 3)

    @test_throws ArgumentError Population(f, (v1, v2); ngen = 100, psze = 30, nslc = 10, nelt = 30)

    p = Population(f, (v, v1, v2), ngen = 100, psze = 30, nslc = 10, nelt = 3)
    @test length(p.vars) == 3
end

@testset "fit_model test" begin
    v = ContinuousVariable(-1f0, 1f0)
    v.vpop = randn(10)
    m = fit_model(v, 1:10)
    @test isa(m, Normal)
    @test isa(rand(v), Float32)

    v = ContinuousBoundedVariable(-1f0, 1f0)
    v.vpop = randn(10)
    m = fit_model(v, 1:10)
    @test isa(m, Truncated)
    @test isa(rand(v), Float32)
end

@testset "update_model test" begin
    v = ContinuousVariable(-1f0, 1f0)
    v.vpop = randn(300)
    update_variable!(v, 5, 1:100, 1:20)
    @test length(v.vpop) == 20
end

@testset "Optimisation" begin
    f(x, y) = -x^2 - y^2

    phst = []
    cb1(p) = push!(phst, values(p))
    bhst = []
    cb2(p) = push!(bhst, optimum(p))
    v1 = ContinuousVariable(-10, 10)
    v2 = ContinuousBoundedVariable(-10, 10)
    p = Population(f, (v1, v2), ngen = 100, psze = 50, nslc = 25, nelt = 5)
    optimise!(p, cbbs = cb1, cbas = cb2)
    (x̂, ŷ), ff = optimum(p)
    @test isapprox(x̂, zero(x̂), atol = eps(typeof(x̂)))
    @test isapprox(ŷ, zero(ŷ), atol = eps(typeof(ŷ)))
    @test isapprox(ff, zero(ff), atol = eps(typeof(ff)))
    @test length(phst) == 100
    @test length(bhst) == 100
    @test neval(p) == 99 * 45 + 50

    v1 = DiscreteVariable(-10:0.1:10)
    v2 = DiscreteBoundedVariable(-10:0.1:10)
    p = Population(f, (v1, v2), ngen = 100, psze = 50, nslc = 25, nelt = 5)
    optimise!(p)
    (x̂, ŷ), ff = optimum(p)
    @test isapprox(x̂, zero(x̂), atol = eps(typeof(x̂)))
    @test isapprox(ŷ, zero(ŷ), atol = eps(typeof(ŷ)))
    @test isapprox(ff, zero(ff), atol = eps(typeof(ff)))

    g(x, y) = x^2 + y^2

    v1 = ContinuousVariable(-10, 10)
    v2 = ContinuousBoundedVariable(-10, 10)
    p = Population(g, (v1, v2), ngen = 100, psze = 50, nslc = 25, nelt = 5, fscl = -1)
    optimise!(p)
    (x̂, ŷ), ff = optimum(p)
    @test isapprox(x̂, zero(x̂), atol = eps(typeof(x̂)))
    @test isapprox(ŷ, zero(ŷ), atol = eps(typeof(ŷ)))
    @test isapprox(ff, zero(ff), atol = eps(typeof(ff)))

    v1 = DiscreteVariable(-10:0.1:10)
    v2 = DiscreteBoundedVariable(-10:0.1:10)
    p = Population(g, (v1, v2), ngen = 100, psze = 50, nslc = 25, nelt = 5, fscl = -1)
    optimise!(p, mt = true)
    (x̂, ŷ), ff = optimum(p)
    @test isapprox(x̂, zero(x̂), atol = eps(typeof(x̂)))
    @test isapprox(ŷ, zero(ŷ), atol = eps(typeof(ŷ)))
    @test isapprox(ff, zero(ff), atol = eps(typeof(ff)))

    f1(x, y) = 0.5 - (sin(sqrt(x^2 + y^2))^2 - 0.5) / (1 + 0.001 * (x^2 + y^2))^2
    v = VariableSet(:x => ContinuousBoundedVariable(-5.12, 5.12), 
                    :y => ContinuousBoundedVariable(-5.12, 5.12))
    p = Population(f1, v, ngen = 10, psze = 50, nslc = 25, nelt = 5)
    optimise!(p)
    (x̂, ŷ), ff = optimum(p)
    @test_throws ArgumentError optimise!(p)
    s = state(p)
    @test s[:fitness][:mean] == mean(p.fpop)
    @test s[:fitness][:std] == std(p.fpop)
    @test s[:x][:mean] == mean(v[:x].modl)
    @test s[:x][:std] == std(v[:x].modl)
    @test s[:y][:mean] == mean(v[:y].modl)
    @test s[:y][:std] == std(v[:y].modl)
end

#rosenbrock(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

#=
A tester :
x Constructeur Population
x build_model
x update_model!
x rand(o::Population)
-

pyplot()
rosenbrock(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

x = -1.5:0.01:2
y = -0.5:0.01:3

z = Surface(rosenbrock, x, y)
surface(x, y, z, color = :jet1, camera = (-30, 30))


p1 = contour(x, y, rosenbrock, fill = true)

X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map(rosenbrock, X, Y)
heatmap(x, y, Z, color=:jet1)

function rosenbrock(x::Vector)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

default(size=(600,600), fc=:heat)
x, y = -1.5:0.1:1.5, -1.5:0.1:1.5
z = Surface((x,y)->rosenbrock([x,y]), x, y)
surface(x,y,z, linealpha = 0.3)

=#
