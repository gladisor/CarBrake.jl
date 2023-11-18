using Dojo
using CarBrake
using ReinforcementLearning
using Distributions
include("../src/environments/car.jl")

struct RandomCarPolicy <: AbstractPolicy
    d::Vector{Uniform}
end

function (policy::RandomCarPolicy)(::CarEnv)
    return rand.(policy.d)
end

policy = RandomCarPolicy(Uniform.(-ones(2), ones(2)))
env = CarEnv(
    max_step = 500,
    control_low = [-0.5, -0.2], control_high = [0.0, 0.2])
reset!(env)

while !is_terminated(env)
    @time env(policy(env), store = true)
    display(reward(env))
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)