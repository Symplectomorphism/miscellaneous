using JuMP
using Mosek, MosekTools

X = 100

best_N = 1
best_t = 0

for N = 1:1:20
# N = 6
    global best_t, best_n

    model = Model(Mosek.Optimizer)
    JuMP.set_silent(model)

    @variables model begin
        t[1:N+1], Int
        a[1:N+1], Int
        s[1:N+1]
        x[1:N,1:3], Bin
    end

    @constraint(model, t[1]==0)
    @constraint(model, t[N+1]==0)
    @constraint(model, a[1]==15)
    @constraint(model, s[1]==21)

    @constraint(model, [i=1:N+1], s[i] >= 0)
    @constraint(model, [i=1:N], sum(x[i,:])==1)

    @constraint(model, sum(t) >= 11)
    @constraint(model, [k=1:N-1], x[k,2] + x[k+1,2] >= 1)

    for k = 1:N
        @constraint(model, t[k+1] - t[k] <= 33 + X*(1-x[k,1]) )
        @constraint(model, t[k+1] - t[k] >= 33 - X*(1-x[k,1]) )

        @constraint(model, t[k+1] - t[k] <= 11 + X*(1-x[k,2]) )
        @constraint(model, t[k+1] - t[k] >= 11 - X*(1-x[k,2]) )

        @constraint(model, t[k+1] - t[k] <= -33 + X*(1-x[k,3]) )
        @constraint(model, t[k+1] - t[k] >= -33 - X*(1-x[k,3]) )


        @constraint(model, s[k+1]  <= s[k]/2 + X*(1-x[k,2]))
        @constraint(model, s[k+1]  >= s[k]/2 - X*(1-x[k,2]) )

        @constraint(model, s[k+1] - s[k] <= -1 + X*(1-x[k,1]-x[k,3])  )
        @constraint(model, s[k+1] - s[k] >= -1 - X*(1-x[k,1]-x[k,3]) )

        @constraint(model, a[k+1]  <= a[k] + X*(1-x[k,1]-x[k,3]) )
        @constraint(model, a[k+1]  >= a[k] - X*(1-x[k,1]-x[k,3]) )

        @constraint(model, a[k+1] - a[k] <= 11 + X*(1-x[k,2]) )
        @constraint(model, a[k+1] - a[k] >= 11 - X*(1-x[k,2]) )
    end

    @objective(model, Max, sum(t))
    optimize!(model)

    max_t = maximum(value.(t))

    if termination_status(model) == MOI.OPTIMAL 
        println("N = $N is feasible.")
        if best_t < max_t
            best_t = maximum(value.(t))
            best_N = N
            println("Maximum t = $best_t")
            println("α + β + γ = $(value.(a)[end] + value.(s)[end] + maximum(value.(t)))")
        end
    end

end