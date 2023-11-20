using Dojo
using ReinforcementLearning
using Flux
using CairoMakie
using CarBrake

include("td3.jl")
# RLBase.update!(app::NeuralNetworkApproximator, gs) = Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)
# Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) = Flux.loadparams!(dest.model, Flux.params(src))

function RLBase.reward(env::CarEnv)
    body = get_body(env.car, :body)
    # return Flux.norm(body.state.x1[1:2] .- env.goal) .- Flux.norm(body.state.x2[1:2] .- env.goal)
    return - Flux.norm(body.state.x2[1:2] .- env.goal) ./ 100.0
end

## control signal: u = [forward power, steering power]
env = CarEnv(
    max_step = 500,
    goal = [0.0, 0.0],
    control_low = [-0.3, -1.0], 
    control_high = [0.3, 1.0])

s_size = length(state(env))
a_size = length(action_space(env))
h_size = 128

actor = NeuralNetworkApproximator(
    Chain(
        Dense(s_size, h_size, relu),
        Dense(h_size, h_size, relu),
        Dense(h_size, a_size, tanh)),
    Adam()
    )

build_critic() = Chain(
    Dense(s_size + a_size, h_size, relu),
    Dense(h_size, h_size, relu),
    Dense(h_size, h_size, relu),
    Dense(h_size, h_size, relu),
    Dense(h_size, 1))

# td3_critic = TD3Critic(build_critic(), build_critic())
# critic = NeuralNetworkApproximator(td3_critic, Adam())
critic = NeuralNetworkApproximator(build_critic(), Adam())

random_policy = RandomPolicy(action_space(env))

ddpg = DDPGPolicy(
    behavior_actor = actor, 
    behavior_critic = critic, 
    target_actor = deepcopy(actor), 
    target_critic = deepcopy(critic),
    start_policy = random_policy,
    act_noise = 1.0,
    na = a_size,
    γ = 0.80f0,
    )

STEPS = env.max_step
EPISODES = 1000

traj = CircularArraySARTTrajectory(
    capacity = STEPS * EPISODES,
    state = Vector{Float64} => size(state(env)),
    action = Vector{Float64} => size(action_space(env)))

agent = Agent(ddpg, traj)
hook = TotalRewardPerEpisode()
run(agent, env, StopAfterEpisode(EPISODES), hook)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, hook.rewards)
save("rewards.png", fig)

reset!(env)
while !is_terminated(env)
    @time env(agent.policy.behavior_actor(state(env)), store = true)
    display(reward(env))
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)

