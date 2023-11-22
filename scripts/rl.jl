using Dojo
using ReinforcementLearning
using Flux
using CairoMakie
using CarBrake
include("td3.jl")

function RLBase.reward(env::CarEnv)
    body = get_body(env.car, :body)
    return Flux.norm(body.state.x1[1:2] .- env.goal) .- Flux.norm(body.state.x2[1:2] .- env.goal)
end

function RLBase.state(env::CarEnv)
    body = get_body(env.car, :body)
    front_axel = get_body(env.car, :front_axel)
    
    fl_wheel = get_body(env.car, :fl_wheel)
    fr_wheel = get_body(env.car, :fr_wheel)
    bl_wheel = get_body(env.car, :bl_wheel)
    br_wheel = get_body(env.car, :br_wheel)

    t = env.step / env.max_step

    return Vector{Float64}(vcat(
        body.state.x2[1:2] ./ 10.0,    ## only care about position in xy
        body.state.v15[1:2],   ## only care about velocity in xy
        Dojo.vector(body.state.q2),
        body.state.ω15,

        # Dojo.vector(front_axel.state.q2),
        # front_axel.state.ω15,    ## how fast is the steering wheel turning

        # Dojo.vector(fl_wheel.state.q2),
        # Dojo.vector(fr_wheel.state.q2),
        # Dojo.vector(bl_wheel.state.q2),
        # Dojo.vector(br_wheel.state.q2),

        # fl_wheel.state.ω15,
        # fr_wheel.state.ω15,
        # bl_wheel.state.ω15,
        # br_wheel.state.ω15,
        t
        ))
    # return get_minimal_state(env.car)
end

BOUNDS = 8.0
function within_bounds(env::CarEnv)
    body = get_body(env.car, :body)
    x = body.state.x2[1]
    y = body.state.x2[2]
    return (-BOUNDS < x < BOUNDS) && (-BOUNDS < y < BOUNDS)
end

PENALTY = -1.0
function RLBase.reward(env::CarEnv)
    body = get_body(env.car, :body)
    r = Flux.norm(body.state.x1[1:2] .- env.goal) .- Flux.norm(body.state.x2[1:2] .- env.goal)

    if !within_bounds(env)
        r += PENALTY
    end

    return r
end

function RLBase.is_terminated(env::CarEnv)
    return !within_bounds(env) || (env.step == env.max_step + 1)
end

function RLBase.state_space(env::CarEnv)
    return Space(state(env))
end