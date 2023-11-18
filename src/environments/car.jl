export CarEnv

mutable struct CarEnv <: AbstractEnv
    car::Mechanism
    control_low::Vector{Float64}
    control_high::Vector{Float64}
    control_to_input::Matrix{Float64}
    step::Int
    max_step::Int
    storage::Storage
end

function CarEnv(;
        max_step::Int = 1000, 
        control_low::Vector = -ones(2),
        control_high::Vector = ones(2))

    car = get_car()
    control_to_input = get_control_to_input(car)
    storage = Storage(max_step, length(car.bodies))

    return CarEnv(
        car, 
        control_low, control_high, control_to_input, 
        1, max_step, storage)
end

function RLBase.reset!(env::CarEnv)
    initialize_car!(env.car, 0.0, -7.0)
    env.storage = Storage(length(env.storage), length(env.car.bodies))
    env.step = 1
end

function RLBase.state(env::CarEnv)
    return get_minimal_state(env.car)
end

function scale_control(env::CarEnv, u::Vector{Float64})
    return (u .+ 1.0) ./ 2.0 .* (env.control_high .- env.control_low) .+ env.control_low
end

"""
Expects a control signal: u ∈ R² ∈ [-1.0, 1.0] x [-1.0, 1.0]

The first component is expected to be the torque to the back wheels
and the second is the torque to the steering wheel.
"""
function (env::CarEnv)(u::Vector{Float64}; store::Bool = false)

    input = env.control_to_input * scale_control(env, u)

    step!(env.car, get_maximal_state(env.car), input)

    if store
        Dojo.save_to_storage!(env.car, env.storage, env.step)
    end

    env.step += 1
end

function RLBase.reward(env::CarEnv)
    x = get_body(env.car, :body).state.x2
    d = sqrt(sum(x[1:2] .^ 2))
    return -d
end

function RLBase.is_terminated(env::CarEnv)
    return env.step == env.max_step + 1
end

function RLBase.action_space(env::CarEnv)
    return Space(
        [
            ClosedInterval(l, h) for (l, h) in zip(env.control_low, env.control_high)
        ])
end