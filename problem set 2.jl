using LinearAlgebra
using Roots
using JuMP
using Ipopt
using Plots

#Problem 1

function iterative_solver(f, x0, α; ϵ=1e-6, maxiter=1000)
    
    x_vals = [x0]  
    residuals = Float64[]  
    flag = 1  
    xn = x0 
    
    for i in 1:maxiter
        # Define g(x) = f(x) + x
        g_x = f(xn) + xn  
        # Update using dampened method
        xn_plus1 = (1 - α) * g_x + α * xn  
        # Compute residual
        residual = abs(xn_plus1 - xn)
        
        # Store the results
        push!(x_vals, xn_plus1)
        push!(residuals, residual)
        
        # Check convergence
        if residual / (1 + abs(xn)) < ϵ
            flag = 0  # Solution found
            break
        end
        
        xn = xn_plus1  # Update for next iteration
    end
    
    g_x = f(xn) + xn 
    
    solution = xn
    f_value = f(solution)
    abs_diff = abs(solution - g_x)
    
    return flag, solution, f_value, abs_diff, x_vals, residuals
end

# function f(x)
f(x) = (x + 1)^(1/3) - x

# initial guess
x0 = 1.0  # Make sure x0 is a float
α = 0.5

result = iterative_solver(f, x0, α)

println("Flag: ", result[1])          # 0 means solution found
println("Solution: ", result[2])      # The solution (root)
println("f(Solution): ", result[3])   # Value of f at the solution
println("Absolute Difference: ", result[4])  # Absolute difference between solution and g(x)
println("x values: ", result[5])      # All x values
println("Residuals: ", result[6])     # All residuals


#Problem 2

function solve_system(α, β)
    A = [
        1 -1 0 α - β β;
        0 1 -1 0 0;
        0 0 1 -1 0;
        0 0 0 1 -1;
        0 0 0 0 1;
    ]
    b = [α, 0, 0, 0, 1]
    
    x_numeric = A \ b
    
    return x_numeric
end

function relative_residual(A, x_exact, b)
    residual = A * x_exact - b  
    return norm(residual) / norm(b) 
end

function condition_number(A)
    return cond(A) 
end

function generate_table(α, β_values)
    println("β\tExact x1\tExact x2\tExact x3\tExact x4\tExact x5\tNumeric x1\tNumeric x2\tNumeric x3\tNumeric x4\tNumeric x5\tRelative Residual\tCondition Number")
    
    for β in β_values
        # exact solution
        exact_solution = [α - β + 1, 1, 1, 1, 1]
        
        # numeric solution
        numeric_solution = solve_system(α, β)
        
        # relative residual
        residual = relative_residual([1 -1 0 α - β β; 0 1 -1 0 0; 0 0 1 -1 0; 0 0 0 1 -1; 0 0 0 0 1], exact_solution, [α, 0, 0, 0, 1])
        
        # condition number
        cond_num = condition_number([1 -1 0 α - β β; 0 1 -1 0 0; 0 0 1 -1 0; 0 0 0 1 -1; 0 0 0 0 1])
        
        println("$β\t$(exact_solution[1])\t$(exact_solution[2])\t$(exact_solution[3])\t$(exact_solution[4])\t$(exact_solution[5])\t$(numeric_solution[1])\t$(numeric_solution[2])\t$(numeric_solution[3])\t$(numeric_solution[4])\t$(numeric_solution[5])\t$residual\t$cond_num")
    end
end

generate_table(0.1, [1, 10, 100, 1000, 10000])
generate_table(0.7, [7, 70, 700, 7000, 70000])

#Problem 3

function NPV(r, C)
    T = length(C) - 1
    npv = 0.0
    for t in 0:T
        npv += C[t+1] / (1 + r)^t
    end
    return npv
end


function internal_rate(C)
    wrapped_NPV(r) = NPV(r, C)
    
    if all(x -> x * C[1] > 0, C)
        return "Warning: No IRR exists (cash flows have the same sign)"
    end

    try
        irr = find_zero(wrapped_NPV, 0.1)  
        return irr
    catch
        return "Warning: Solver failed to find a root"
    end
end



cash_flows = [-5, 0, 0, 2.5, 5]
irr = internal_rate(cash_flows)
println("IRR: ", irr)

cash_flows_2 = [-10, 3, 3, 3, 3]
irr_2 = internal_rate(cash_flows_2)
println("IRR: ", irr_2)

cash_flows_3 = [5, 5, 5, 5, 5]
irr_3 = internal_rate(cash_flows_3)
println("IRR: ", irr_3)


#Problem 4
using JuMP
using Ipopt
using Plots

function ces_production(x1, x2, α, σ)
    return (α * x1^(σ - 1) + (1 - α) * x2^(σ - 1))^(1 / (σ - 1))
end

function minimize_cost(α, σ, w1, w2, y)
    model = Model(Ipopt.Optimizer)

    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    @objective(model, Min, w1 * x1 + w2 * x2)

    @constraint(model, ces_production(x1, x2, α, σ) == y)

    optimize!(model)

    optimal_x1 = value(x1)
    optimal_x2 = value(x2)

    return optimal_x1, optimal_x2
end

function plot_cost_function_and_demand(α, sigma_values, w2, y)
    w1_range = 0.1:0.1:10 

    for σ in sigma_values
        cost = Float64[]
        x1_demand = Float64[]
        x2_demand = Float64[]

        for w1 in w1_range
            opt_values = minimize_cost(α, σ, w1, w2, y)
            push!(cost, w1 * opt_values[1] + w2 * opt_values[2])
            push!(x1_demand, opt_values[1])
            push!(x2_demand, opt_values[2])
        end

        plot(w1_range, cost, label="Cost", xlabel="w1", ylabel="Cost", title="Cost and Demand (σ = $σ)")
        plot!(w1_range, x1_demand, label="x1 Demand")
        plot!(w1_range, x2_demand, label="x2 Demand")
    end
end

plot_cost_function_and_demand(0.5, [0.25, 1, 4], 1.0, 1.0)
