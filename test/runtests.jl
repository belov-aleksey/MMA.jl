using .MMA
using Test

import .MMA: eval_constraint, eval_objective, dim
import LinearAlgebra: norm

# [1] https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/
# [2] Luo Y. An enhanced aggregation method for topology optimization with local
# stress constraints [Text] / Y. Luo, M. Y. Wang, Z. Kang // Computer Methods in
# Applied Mechanics and Engineering. – 2013. – Vol. 254. – p. 31-41.
# [3] Christensen P.W. An introduction to structural optimization [Text]/ P. W.
# Christensen, A. Klarbring. – Berlin: Springer, 2009. – p. 64-65.

@testset "NLopt.jl tutorial example [1]" begin
    function f(x::Vector, grad::Vector)
        if length(grad) != 0
            grad[1] = 0.0
            grad[2] = 0.5/sqrt(x[2])
        end
        sqrt(x[2])
    end

    function g(x::Vector, grad::Vector, a, b)
        if length(grad) != 0
            grad[1] = 3a * (a*x[1] + b)^2
            grad[2] = -1
        end
        (a*x[1] + b)^3 - x[2]
    end

    ndim = 2
    m = MMAModel(ndim, f, xtol = 1e-6, store_trace=true)

    box!(m, 1, 0.0, 10.0)
    box!(m, 2, 0.0, 10.0)

    ineq_constraint!(m, (x,grad) -> g(x,grad,2,0))
    ineq_constraint!(m, (x,grad) -> g(x,grad,-1,1))


    ################3

    @test dim(m) == 2

    # Objective
    let
        grad1 = zeros(2)
        grad2 = zeros(2)
        p = [1.234, 2.345]
        @test eval_objective(m, p, grad2) ≈ f(p, grad1)
        @test grad1 ≈ grad2
    end

    # Box
    @test min(m, 1) == 0.0
    @test max(m, 1) == 10.0
    @test min(m, 2) == 0.0
    @test max(m, 2) == 10.0

    # Inequalities
    let
        grad1 = zeros(2)
        grad2 = zeros(2)
        p = [1.234, 2.345]
        @test eval_constraint(m, 1, p, grad1) ≈ g(p,grad2 ,2,0)
        @test grad1 ≈ grad2

        @test eval_constraint(m, 2, p, grad1) ≈ g(p,grad2,-1,1)
        @test grad1 ≈ grad2
    end

    r = optimize(m, [1.234, 2.345])
    @test abs(r.minimum - sqrt(8/27)) < 1e-6
    @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
end

@testset "Example from [2]" begin
    function objective_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 1.0
            grad[2] = 1.0
        end
        return x[1] + x[2]
    end

    function constraint_fn1(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = -x[1] * x[2] / 10
            grad[2] = -x[1]^2 / 20
        end
        return 1 - x[1]^2 * x[2] / 20
    end

    function constraint_fn2(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 1 / 15 * (5 - x[1] - x[2]) + 1 / 60 * (12 - x[1] + x[2])
            grad[2] = 1 / 15 * (5 - x[1] - x[2]) + 1 / 60 * (-12 + x[1] - x[2])
        end
        return 1 - (x[1] + x[2]-5)*(x[1]+x[2]-5)/30-(x[1]-x[2]-12)*(x[1]-x[2]-12)/120
    end

    function constraint_fn3(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = (160 * x[1]) / ((5 + x[1]^2 + 8 * x[2]) * (5 + x[1]^2 + 8 * x[2]))
            grad[2] =  640 / ((5 + x[1]^2 + 8 * x[2]) * (5 + x[1]^2 + 8 * x[2]))
        end
        return 1 - 80 / (x[1]^2 + 8 * x[2] + 5)
    end

    function constraint_fn4(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = -1
            grad[2] =  (2 * x[2]) / 15
        end
        return -x[1] + x[2]^2 / 15 + 7 / 15
    end

    function constraint_fn5(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 1
            grad[2] =  (2 * x[2]) / 15
        end
        return -(1 / 5)
    end

    ndim = 2
    m = MMAModel(ndim, objective_fn, xtol = 1e-7, store_trace=true)
    box!(m, 1, 0.0, 10.0)
    box!(m, 2, 0.0, 10.0)

    ineq_constraint!(m, (x,grad) -> constraint_fn1(x,grad))
    ineq_constraint!(m, (x,grad) -> constraint_fn2(x,grad))
    ineq_constraint!(m, (x,grad) -> constraint_fn3(x,grad))
    ineq_constraint!(m, (x,grad) -> constraint_fn4(x,grad))
    ineq_constraint!(m, (x,grad) -> constraint_fn5(x,grad))

    r = optimize(m, [5.0, 5.0])
    @test abs(r.minimum - 5.176474756) < 1e-4
    @test norm(r.minimizer - [3.11418, 2.0623]) < 5e-4
end


@testset "Example from [3]" begin
    function objective_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = 1.0
            grad[2] = 1.0
        end
        return x[1] + x[2]
    end

    function constraint_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = -128/(16*x[1]+9*x[2])^2+40.5/(9*x[1]+16*x[2])^2
            grad[2] = -72/(16*x[1]+9*x[2])^2+72/(9*x[1]+16*x[2])^2
        end
        return 8/(16*x[1] + 9*x[2]) - 4.5/(9*x[1] + 16*x[2]) - 0.1
    end

    ndim = 2
    m = MMAModel(ndim, objective_fn, xtol = 1e-6, store_trace=true)

    box!(m, 1, 0.2, 2.5)
    box!(m, 2, 0.2, 2.5)

    ineq_constraint!(m, (x,grad) -> constraint_fn(x,grad))

    r = optimize(m, [2.0, 1.0])
    @test abs(r.minimum - 1.074910) < 2e-4
    @test norm(r.minimizer - [0.87491, 0.2]) < 2e-4
end