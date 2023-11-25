using Dojo, ReinforcementLearning
using CarBrake

env = CarEnv(
    max_step = 1000,
    goal = [0.0, 3.0],
    control_low = [-0.2, -0.1],
    control_high = [0.2, 0.1]
    )

policy = RandomPolicy(action_space(env))

reset!(env)
while !is_terminated(env)
    @time env(policy(env), store = true)
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)