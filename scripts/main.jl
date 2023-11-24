using Dojo, ReinforcementLearning
using CarBrake

env = CarEnv(
    max_step = 1000,
    goal = [0.0, 3.0],
    control_low = [-0.2, -0.1],
    control_high = [0.2, 0.1]
    )

# u = [1.0, -1.0]
# input = env.control_to_input * CarBrake.scale_control(env, u)

# display(maximum(input))
# display(minimum(input))

policy = RandomPolicy(action_space(env))

reset!(env)
while !is_terminated(env)
    steer = sin(4pi*env.step / env.max_step)
    display(steer)
    @time env([-1.0, steer], store = true)
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)