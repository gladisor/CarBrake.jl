using Dojo
using CarBrake
using ReinforcementLearning
include("../src/environments/car.jl")

env = CarEnv()

while !is_terminated(env)
    env(rand(action_space(env)))
    display(reward(env))
end
