using Random
using PyPlot
################################################################################
#
# *********************************
# * (1,5) (2,5) (3,5) (4,5) (5,5) *
# * (1,4) (2,4) (3,4) (4,4) (5,4) *
# * (1,3) (2,3) (3,3) (4,3) (5,3) *
# * (1,2) (2,2) (3,2) (4,2) (5,2) *
# * (1,1) (2,1) (3,1) (4,1) (5,1) *
# *********************************
#
################################################################################
 
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

mutable struct MonteCarlo
    e::Env
    p::Array{Matrix{Float64}, 1}
    counts::Array{Int, 1}
    avg_cost::Float64
    fig::Figure
end

function MonteCarlo()
    p = Array{Matrix{Float64}, 1}()
    for i = 1:9
        push!(p, zeros(5,5))
    end
    counts = zeros(Int, 9)
    fig = figure(1)
    MonteCarlo(Env(), p, counts, 0.0, fig)
end

function reset!(m::MonteCarlo)
    reset_environment!(m.e)
    for i = 1:9
        m.p[i] = zeros(Float64, 5, 5)
        m.counts[i] = 0
    end
    m.avg_cost = 0.0
    m.fig.clf()
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


function monte_carlo(m::MonteCarlo=MonteCarlo(); iter::Int=1_000_000, t::Float64=0.01)
    reset!(m)
    for n = 1:iter
        reset_environment!(m.e)
        simulate(m.e)
        @assert m.e.state_history[end] == [5,5]
        for i = 1:length(m.e.state_history)
            for j = 1:5
                for k = 1:5
                    if m.e.state_history[i] == [j,k]
                        m.p[i][j,k] += 1
                    end
                end
            end
        end
        m.counts[1:length(m.e.state_history)] .+= 1
        m.avg_cost = m.avg_cost * (n-1)/n + m.e.cost/n
    end
    for i = 1:9
        # m.p[i] = m.p[i] ./ m.counts[i]
        m.p[i] = m.p[i] ./ m.counts[1]
    end
    p_hoeff = 2*exp( -2*iter*iter*t*t / ((8-4)^2) / iter )
    @info "Average cost of $iter samples is C̄ = $(round(m.avg_cost, digits=4))."
    @info "By the Hoeffding bound, P(|C̄ - E[C]| >= $t) <=  $(round(p_hoeff, digits=6))."

    show_distribution(m)
end

function show_distribution(m::MonteCarlo)
    x = 0:5
    y = 6:-1:1
    for i = 1:8
        ax = m.fig.add_subplot(2,4,i)
        # ax.pcolormesh(x, y, m.p[i+1])
        ax.imshow(m.p[i+1] |> rotl90, cmap="YlGn")
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)

        for j = 1:5
            for k = 5:-1:1
                ax.text(k-1, j-1, rotl90(m.p[i+1])[j,k], ha="center", va="center", color="w")
            end
        end
        # ax.set_xlabel("Hamle: $(i)", fontsize=16)
        ax.set_xlabel("Move: $(i)", fontsize=16)
    end
    # m.fig.suptitle("Durum Dağılımı", fontsize=16)
    m.fig.suptitle("State Distribution", fontsize=16)
    m.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
end