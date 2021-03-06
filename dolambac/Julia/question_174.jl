using JuMP
using Mosek, MosekTools
using LinearAlgebra
using DataFrames

function find_best(best_N::Int=0, best_t::Int=0)
    best_value = 0.0
    best_history = DataFrame()
    for N = 0:Int(round(log(2,21), RoundUp)*2)
        X = 100

        model = Model(Mosek.Optimizer)
        JuMP.set_silent(model)

        @variables model begin
            t[1:N+1], Int       # The time at step k ∈ [1, N].
            a[1:N+1], Int       # The age of the lady at step k ∈ [1, N].
            s[1:N+1]            # The amount of resources available at step k ∈ [1, N].
            x[1:N,1:3], Bin     # 3 actions are available at step k ∈ [1, N]: 
                                # Go forward by 33 years, Wait for 11 years, or
                                # Go backward by 33 years.
            y[1:N+1], Bin       # Auxiliary binary variable to maximize the maximum element of t.
            T, Int
        end

        @constraint(model, t[1]==0)                             # Time starts at 0
        @constraint(model, t[N+1]==0)                           # Final time must be back to 0
        @constraint(model, a[1]==15)                            # Initial age
        @constraint(model, s[1]==21)                            # Initial resource amount

        @constraint(model, [i=1:N], s[i] >= 1)                  # She must have at least one resource left to travel
        @constraint(model, [i=1:N], sum(x[i,:])==1)             # Only one decision is allowed at each step

        @constraint(model, [k=1:N-1], x[k,2] + x[k+1,2] >= 1)   # She must wait at least for one step before traveling again.

        @constraint(model, sum(y) == 1)                         # We want to maximize only one element of t[k], y is an
                                                                # indicator which must have only one element to be 1 and
                                                                # other elements will be 0.
        @constraint(model, [k=1:N+1], T - t[k] <= X*(1-y[k]))   # If y[j] = 1 then T must be less than t[j] 
                                                                # We will maximize T so that the t[j] increases; but 
                                                                # t[j] is supposed to be the maximum of t by the y 
                                                                # construction b/c is y is such that t[k] for k =/= j 
                                                                # was maximum then we cannot maximize T as much as 
                                                                # possible, so the mixed integer program will switch 
                                                                # y[k] = 1 to y[j] = 1!

        for k = 1:N
            # Depending on the action x[k,?] taken at step k ∈ [1, N], model the value of the next time t.
            @constraint(model, t[k+1] - t[k] <= 33 + X*(1-x[k,1]) )
            @constraint(model, t[k+1] - t[k] >= 33 - X*(1-x[k,1]) )

            @constraint(model, t[k+1] - t[k] <= 11 + X*(1-x[k,2]) )
            @constraint(model, t[k+1] - t[k] >= 11 - X*(1-x[k,2]) )

            @constraint(model, t[k+1] - t[k] <= -33 + X*(1-x[k,3]) )
            @constraint(model, t[k+1] - t[k] >= -33 - X*(1-x[k,3]) )

            # Depending on the action x[k,?] taken at step k ∈ [1, N], 
            # model the value of the amount of available resources at the next step.
            @constraint(model, s[k+1]  <= s[k]/2 + X*(1-x[k,2]))
            @constraint(model, s[k+1]  >= s[k]/2 - X*(1-x[k,2]) )

            @constraint(model, s[k+1] - s[k] <= -1 + X*(1-x[k,1]-x[k,3])  )
            @constraint(model, s[k+1] - s[k] >= -1 - X*(1-x[k,1]-x[k,3]) )

            # Depending on the action x[k,?] taken at step k ∈ [1, N], compute the age of the lady.
            @constraint(model, a[k+1]  <= a[k] + X*(1-x[k,1]-x[k,3]) )
            @constraint(model, a[k+1]  >= a[k] - X*(1-x[k,1]-x[k,3]) )

            @constraint(model, a[k+1] - a[k] <= 11 + X*(1-x[k,2]) )
            @constraint(model, a[k+1] - a[k] >= 11 - X*(1-x[k,2]) )
        end

        @objective(model, Max, T)      # Linear programming proxy for maximizing max(t).
        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL 
            println("N = $N is feasible.")
            if best_t < value(T)
                best_t = Int(value(T))
                best_N = Int(round(N))
                best_value = value.(a)[end] + value.(s)[end] + round(maximum(value.(t)))
                best_history = DataFrame()
                insertcols!(best_history, 1, :age=>Int.(round.(value.(a))))
                insertcols!(best_history, 2, :resource=>value.(s))
                insertcols!(best_history, 3, :time=>Int.(round.(value.(t))))
            end
            println("Maximum t = $(Int(value(T)))")
            println("α + β + γ = $(round(value.(a)[end] + value.(s)[end] + value(T); digits=6))")
            println()
        end
    end

    return best_t, best_N, best_value, best_history
end
best_t, best_N, best_value, best_history = find_best()

println("The best path to take has the value: α + β + γ = $(best_value)")
println("Its trajectory is (age, resource, time):")
display(best_history)