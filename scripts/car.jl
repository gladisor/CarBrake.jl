using ReinforcementLearning
using Dojo
using Distributions, IntervalSets, Flux
using CarBrake

env = CarEnv(
    goal = [7.0, 7.0],
    control_low = [-0.2, -0.1], 
    control_high = [0.2, 0.1],
    max_step = 100
    )

policy = RandomPolicy(action_space(env))

while !is_terminated(env)
    @time env(policy(env), store = true)
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)