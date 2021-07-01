using Random

struct Env
    # board::Matrix
    state_history::Array{Array{Int, 1}}
    action_history::Array{Symbol, 1}
    goal::Array{Int, 1}
    true_keymap::Dict{Int, Symbol}
    belief_keymap::Dict{Int, Symbol}
    cost::Int       # equal to total number of moves
end

function Env()
    # board = Matrix()
    s = Array{Array{Int, 1}}(undef, 0)
    a = Array{Array{Int, 1}}(undef, 0)
    push!(s, [2,2])
    g = [5,5]

    key_set = randperm(4)
    action_set = [:left, :right, :up, :down]
    true_keymap = Dict(zip(key_set, action_set))
    belief_keymap = Dict{Int, Symbol}()

    Env(s, a, g, true_keymap, belief_keymap, 0)
end


function dynamics(s::Vector, a::Symbol)
    if a == :left
        s[1] != 0 ? s_next = s - [1, 0] : s_next = s
    elseif a == :right
        s[1] != 5 ? s_next = s + [1, 0] : s_next = s
    elseif a == :up
        s[2] != 5 ? s_next = s + [0, 1] : s_next = s
    elseif a==:down
        s[2] != 0 ? s_next = s - [0, 1] : s_next = s
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
        if e.state_history[end] == [2,3] || e.state_history[end] == [3,2]
            push!(e.action_history, e.belief_keymap[1])
            push!(e.state_history, dynamics(e.state_history[1], e.action_history[end]))
            e.cost += 1

            
        end
    end
end