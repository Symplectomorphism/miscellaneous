using Random

mutable struct Env
    state_history::Array{Array{Int, 1}}
    action_history::Array{Symbol, 1}
    goal::Array{Int, 1}
    true_keymap::Dict{Int, Symbol}
    belief_keymap::Dict{Int, Symbol}
    cost::Int       # equal to total number of moves
end

function Env()
    s = Array{Array{Int, 1}}(undef, 0)
    a = Array{Array{Int, 1}}(undef, 0)
    push!(s, [3,3])
    g = [5,5]

    key_set = randperm(4)
    action_set = [:left, :right, :up, :down]
    true_keymap = Dict(zip(key_set, action_set))
    belief_keymap = Dict{Int, Symbol}()

    Env(s, a, g, true_keymap, belief_keymap, 0)
end

function reset_environment!(e::Env)
    empty!(e.state_history)
    push!(e.state_history, [3,3])
    empty!(e.action_history)
    key_set = randperm(4)
    action_set = [:left, :right, :up, :down]
    e.true_keymap = Dict(zip(key_set, action_set))
    e.belief_keymap = Dict{Int, Symbol}()
    e.cost = 0
end


function dynamics(s::Vector, a::Symbol)
    if a == :left
        s[1] != 1 ? s_next = s - [1, 0] : s_next = s
    elseif a == :right
        s[1] != 5 ? s_next = s + [1, 0] : s_next = s
    elseif a == :up
        s[2] != 5 ? s_next = s + [0, 1] : s_next = s
    elseif a==:down
        s[2] != 1 ? s_next = s - [0, 1] : s_next = s
    else
        error("Not a valid action")
    end
    return s_next
end

function simulate(e::Env)
    if e.cost == 0
        push!(e.action_history, e.true_keymap[1])
        push!(e.belief_keymap, 1=>e.action_history[1])
        push!(e.state_history, dynamics(e.state_history[1], e.action_history[1]))
        e.cost += 1
    end
    if e.cost == 1
        if e.state_history[end] == [4,3]
            push!(e.action_history, e.belief_keymap[1])
            push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
            e.cost += 1

            push!(e.action_history, e.true_keymap[2])
            push!(e.belief_keymap, 2=>e.action_history[end])
            push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
            e.cost += 1
            if e.state_history[end] == [5,4]
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1
            end
            if e.state_history[end] == [5,2]
                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [5,3]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [4,2]
                    push!(e.action_history, e.belief_keymap[1])
                    push!(e.belief_keymap, 4=>:up)
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
            if e.state_history[end] == [4,3]
                push!(e.action_history, e.belief_keymap[1])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [5,2]
                    push!(e.belief_keymap, 4=>:up)

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [5,4]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
        end
    end
    if e.cost == 1
        if e.state_history[end] == [3,4]
            push!(e.action_history, e.belief_keymap[1])
            push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
            e.cost += 1

            push!(e.action_history, e.true_keymap[2])
            push!(e.belief_keymap, 2=>e.action_history[end])
            push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
            e.cost += 1
            if e.state_history[end] == [4,5]
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1
            end
            if e.state_history[end] == [2,5]
                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [3,5]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [2,4]
                    push!(e.action_history, e.belief_keymap[1])
                    push!(e.belief_keymap, 4=>:right)
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
            if e.state_history[end] == [3,4]
                push!(e.action_history, e.belief_keymap[1])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [2,5]
                    push!(e.belief_keymap, 4=>:right)

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [4,5]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
        end
    end

    if e.cost == 1
        if e.state_history[end] == [3,2]
            push!(e.action_history, e.true_keymap[2])
            push!(e.belief_keymap, 2=>e.action_history[end])
            push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
            e.cost += 1

            if e.state_history[end] == [2,2]
                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [3,2]
                    push!(e.belief_keymap, 3=>:right)
                    push!(e.belief_keymap, 4=>:up)

                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                else
                    push!(e.belief_keymap, 3=>:up)
                    push!(e.belief_keymap, 4=>:right)

                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
            if e.state_history[end] == [3,3]
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [4,5]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [2,5]
                    push!(e.belief_keymap, 4=>:right)

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
            if e.state_history[end] == [4,2]
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [5,3]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [4,2]
                    push!(e.belief_keymap, 4=>:up)
                    push!(e.action_history, e.belief_keymap[2])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
        end
    end
    if e.cost == 1
        if e.state_history[end] == [2,3]
            push!(e.action_history, e.true_keymap[2])
            push!(e.belief_keymap, 2=>e.action_history[end])
            push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
            e.cost += 1

            if e.state_history[end] == [2,2]
                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [3,2]
                    push!(e.belief_keymap, 3=>:right)
                    push!(e.belief_keymap, 4=>:up)

                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                else
                    push!(e.belief_keymap, 3=>:up)
                    push!(e.belief_keymap, 4=>:right)

                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
            if e.state_history[end] == [3,3]
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [5,4]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [5,2]
                    push!(e.belief_keymap, 4=>:up)

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
            if e.state_history[end] == [2,4]
                push!(e.action_history, e.belief_keymap[2])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                push!(e.action_history, e.true_keymap[3])
                push!(e.belief_keymap, 3=>e.action_history[end])
                push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                e.cost += 1

                if e.state_history[end] == [3,5]
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[3])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
                if e.state_history[end] == [2,4]
                    push!(e.belief_keymap, 4=>:right)
                    push!(e.action_history, e.belief_keymap[2])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1

                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                    push!(e.action_history, e.belief_keymap[4])
                    push!(e.state_history, dynamics(e.state_history[end], e.action_history[end]))
                    e.cost += 1
                end
            end
        end
    end
end


function monte_carlo(;iter::Int=1_000_000, t::Float64=0.01)
    e = Env()
    average_cost = 0.0
    for n = 1:iter
        reset_environment!(e)
        simulate(e)
        @assert e.state_history[end] == [5,5]
        average_cost = average_cost*(n-1)/n + e.cost/n
    end
    p = 2*exp( -2*iter*iter*t*t / ((8-4)^2) / iter )
    @info "Average cost of $iter samples is C̄ = $(round(average_cost, digits=4))."
    @info "By the Hoeffding bound, P(|C̄ - E[C]| >= $t) <=  $(round(p, digits=6))."
    return average_cost
end