include("rl.jl")
Flux.device!(2)

## control signal: u = [forward power, steering power]
env = CarEnv(
    max_step = 1000,
    goal = [0.0, 3.0],
    control_low = [-0.3, -1.0],
    control_high = [0.3, 1.0])

env = MultiThreadEnv(() -> deepcopy(env), 5)

s_size = length(state(env[1]))
a_size = length(action_space(env[1]))
h_size = 128

activation = relu
build_actor() = Chain(
    Dense(s_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, a_size, tanh))

build_critic() = Chain(
    Dense(s_size + a_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, 1))

actor = NeuralNetworkApproximator(build_actor(), Adam(1e-4))
critic = NeuralNetworkApproximator(build_critic(), Adam(1e-3))

ddpg = DDPGPolicy(
    behavior_actor = actor |> gpu,
    behavior_critic = critic |> gpu,
    target_actor = deepcopy(actor) |> gpu,
    target_critic = deepcopy(critic) |> gpu,
    start_policy = RandomPolicy(action_space(env)) |> gpu,
    na = a_size,
    act_noise = 1.0,
    γ = 0.99f0,
    ρ = 0.999f0,
    batch_size = 64
    )

# hook = TotalRewardPerEpisode()
hook = TotalBatchRewardPerEpisode(5)

STEPS = env[1].max_step
EPISODES = 50 #2000

traj = CircularArraySARTTrajectory(
    capacity = 1000000,
    state = Vector{Float64} => size(state(env[1])),
    action = Vector{Float64} => size(action_space(env[1])))

agent = Agent(ddpg, traj)
# run(agent, env, StopAfterEpisode(EPISODES), hook)
run(agent, env, StopAfterNSeconds(5.0), hook)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, hook.rewards)
save("rewards.png", fig)

# # # reset!(env)
# # # while !is_terminated(env)
# # #     @time env(agent.policy.behavior_actor(state(env)), store = true)
# # # end

# # # vis = Visualizer()
# # # open(vis)
# # # visualize(env.car, env.storage, vis = vis)

