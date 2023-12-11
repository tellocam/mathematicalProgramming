using JuMP
using GLPK

struct Node
    lb::Vector{Float64}
    ub::Vector{Float64}
    parent::Union{Nothing, Int}
end

function solve_lp(c, A, b, lb, ub)
    model = Model(GLPK.Optimizer)  # Set time limit to 60 seconds

    @variable(model, lb[i] <= x[i=1:length(c)] <= ub[i])
    @objective(model, Max, sum(c[i] * x[i] for i in 1:length(c)))
    @constraint(model, con, A * x .<= b)

    optimize!(model)

    x_values = JuMP.value.(x)
    obj_value = objective_value(model)

    return x_values, obj_value
end

function branch_and_bound(c, A, b, lb, ub)
    pending_nodes = [Node(lb, ub, nothing)]

    best_solution = nothing
    best_value = -Inf

    while !isempty(pending_nodes)
        node = popfirst!(pending_nodes)
        lb, ub = node.lb, node.ub

        println("Exploring node with lb = $lb, ub = $ub")

        if any(ub[i] < lb[i] for i in 1:length(c))
            println("Infeasible node, skipping.")
            continue  # Infeasible node
        end

        if sum(c[i] * (ub[i] - lb[i]) for i in 1:length(c)) <= best_value
            println("Pruning node, skipping.")
            continue  # Prune the node
        end

        # Solve LP
        x, obj_value = solve_lp(c, A, b, lb, ub)

        println("Solved LP with x = $x, obj_value = $obj_value")

        if obj_value > best_value
            best_solution = x
            best_value = obj_value
            println("Updating best solution: $best_solution, best_value: $best_value")
        end

        # Identify fractional variables
        frac_values = abs.(x - round.(x))
        max_frac_idx = argmax(frac_values)

        if all(abs.(x[i] - round(x[i])) < 1e-6 for i in 1:length(x))
            println("Integer solution found.")
            println("Variable values: ", x)
            println("Rounded values: ", round.(x))
            continue  # Integer solution found
        end

        # Branch on the variable with the largest fractional part
        idx_to_branch = max_frac_idx
        lb_branch = copy(lb)
        ub_branch = copy(ub)

        ub_branch[idx_to_branch] = floor(x[idx_to_branch])
        println("Branching down on variable $idx_to_branch, new ub = $ub_branch")
        push!(pending_nodes, Node(lb, ub_branch, length(pending_nodes)))

        lb_branch[idx_to_branch] = ceil(x[idx_to_branch])
        println("Branching up on variable $idx_to_branch, new lb = $lb_branch")
        push!(pending_nodes, Node(lb_branch, ub, length(pending_nodes)))
    end

    return best_solution, best_value
end

# Define the problem
c = [2, 4, 1]
A = [1 3 2; 3 5 1]
b = [12, 16]
lb = zeros(3)
ub = [Inf, 2, Inf]

# Initial branch on x2 <= 2
ub[2] = 2

# Solve the problem using Branch-and-Bound
solution, value = branch_and_bound(c, A, b, lb, ub)

println("Optimal Solution: ", solution)
println("Optimal Value: ", value)