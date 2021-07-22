using JuMP
using Mosek, MosekTools
using LinearAlgebra

function find_best(best_N::Int=0, best_t::Int=0)
    best_value = 0.0
    for N = 0:Int(round(log(2,21), RoundUp)*2)
        X = 100

        model = Model(Mosek.Optimizer)
        JuMP.set_silent(model)

        @variables model begin
            t[1:N+1], Int       # The time at step k ∈ [1, N].
            a[1:N+1], Int       # The age of the guy at step k ∈ [1, N].
            s[1:N+1]            # The amount of resources available at step k ∈ [1, N].
            x[1:N,1:3], Bin     # 3 actions are available at step k ∈ [1, N]: 
                                # Go forward by 33 years, Wait for 11 years, or
                                # Go backward by 11 years.
            y[1:N+1], Bin       # Auxiliary binary variable to maximize the maximum element of t.
            T, Int
        end

        @constraint(model, t[1]==0)                             # time starts at 0
        @constraint(model, t[N+1]==0)                           # final time must be back to 0
        @constraint(model, a[1]==15)                            # initial age
        @constraint(model, s[1]==21)                            # initial resource amount

        @constraint(model, [i=1:N], s[i] >= 1)                  # he must have at least one element left to travel
        @constraint(model, [i=1:N], sum(x[i,:])==1)             # only one decision is allowed at each step

        @constraint(model, [k=1:N-1], x[k,2] + x[k+1,2] >= 1)   # he must wait at least for one step before traveling again.
        @constraint(model, sum(y) == 1)                         # if y[k] = 1 then T must be less than t[k]
        @constraint(model, [k=1:N+1], T - t[k] <= X*(1-y[k]))   # We maximize T so that the largest of t[k] increases.

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

            # Depending on the action x[k,?] taken at step k ∈ [1, N], compute the age of the guy.
            @constraint(model, a[k+1]  <= a[k] + X*(1-x[k,1]-x[k,3]) )
            @constraint(model, a[k+1]  >= a[k] - X*(1-x[k,1]-x[k,3]) )

            @constraint(model, a[k+1] - a[k] <= 11 + X*(1-x[k,2]) )
            @constraint(model, a[k+1] - a[k] >= 11 - X*(1-x[k,2]) )
        end

        @objective(model, Max, T)      # Linear programming proxy for maximizing max(t).
        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL 
            println("N = $N is feasible.")
            display( hcat(value.(a), value.(s), value.(t)) )
            if best_t < value(T)
                best_t = Int(value(T))
                best_N = Int(round(N))
                best_value = value.(a)[end] + value.(s)[end] + round(maximum(value.(t)))
            end
            println("Maximum t = $(Int(value(T)))")
            println("α + β + γ = $(value.(a)[end] + value.(s)[end] + round(value(T)))")
            println()
        end
    end

    return best_t, best_N, best_value
end
best_t, best_N, best_value = find_best()