export CarEnv

mutable struct CarEnv <: AbstractEnv
    car::Mechanism
    control_low::Vector{Float64}
    control_high::Vector{Float64}
    control_to_input::Matrix{Float64}

    goal::Vector{Float64}
    step::Int
    max_step::Int
    storage::Storage
end

function reset_car(goal::Vector)
    car = get_car()
    car = add_goal(car, goal)
    initialize_car!(
        car,
        rand(Uniform(-7.0, 7.0)),
        -7.0
        )

    return car
end

function CarEnv(;
        goal::Vector,
        max_step::Int = 1000, 
        control_low::Vector = [-0.2, -0.1],
        control_high::Vector = [0.2, 0.1])

    car = reset_car(goal)
    control_to_input = get_control_to_input(car)
    storage = Storage(max_step, length(car.bodies))

    return CarEnv(
        car, 
        control_low, 
        control_high, 
        control_to_input, 
        goal, 1, max_step, storage)
end

function RLBase.reset!(env::CarEnv)
    env.car = reset_car(env.goal)
    env.storage = Storage(length(env.storage), length(env.car.bodies))
    env.step = 1
end

function RLBase.state(env::CarEnv)
    body = get_body(env.car, :body)
    front_axel = get_body(env.car, :front_axel)
    t = env.step / env.max_step

    return Vector{Float64}(vcat(
        body.state.x2[1:2] ./ 10.0,    ## only care about position in xy
        body.state.v15[1:2],   ## only care about velocity in xy
        Dojo.vector(body.state.q2),
        body.state.ω15,
        Dojo.vector(front_axel.state.q2),
        front_axel.state.ω15,    ## how fast is the steering wheel turning
        t
        ))
end

function scale_control(env::CarEnv, u::Vector{Float64})
    return (u .+ 1.0) ./ 2.0 .* (env.control_high .- env.control_low) .+ env.control_low
end

"""
Expects a control signal: u ∈ R² ∈ [-1.0, 1.0] x [-1.0, 1.0]

The first component is expected to be the torque to the back wheels
and the second is the torque to the steering wheel.
"""
function (env::CarEnv)(u::AbstractArray{Float64}; store::Bool = false)

    if u isa SubArray
        u = Vector{Float64}(u)
    end

    input = env.control_to_input * scale_control(env, u)

    step!(env.car, get_maximal_state(env.car), input)

    if store
        Dojo.save_to_storage!(env.car, env.storage, env.step)
    end

    env.step += 1
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

function RLBase.action_space(::CarEnv)
    return Space([ClosedInterval(-1.0, 1.0) for i in 1:2])
end