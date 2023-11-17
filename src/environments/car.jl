# mutable struct CarEnv <: AbstractEnv
#     mech::Mechanism
#     control_mask::Matrix
#     goal::Vector{Float64}
#     step::Int
#     max_step::Int
# end

# function CarEnv(goal = [10.0, 10.0], max_step = 100)
#     mech = get_car()
#     initialize_car!(mech)
#     actions = CarBrake.car_actions()
#     return CarEnv(mech, actions, goal, 0, max_step)
# end

# RLBase.is_terminated(env::CarEnv) = env.step == env.max_step
# RLBase.state(env::CarEnv) = get_minimal_state(env.mech)
# RLBase.state_space(env::CarEnv) = state(env)
# RLBase.action_space(env::CarEnv) = 1:size(env.actions, 2)

# function RLBase.reset!(env::CarEnv)
#     initialize_car!(env.mech)
#     env.step = 0
# end

# function (env::CarEnv)(action::Int)
#     u = env.actions[:, action]
#     step!(env.mech, get_maximal_state(env.mech), u)
#     env.step += 1
# end

# function RLBase.reward(env::CarEnv)
#     x = get_body(env.mech, :body).state.x2
#     d = sqrt(sum((env.goal .- x[1:2]) .^ 2))
#     return d
# end