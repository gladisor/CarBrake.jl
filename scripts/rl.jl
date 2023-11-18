using Dojo
using ReinforcementLearning
using Flux
using IntervalSets
using CarBrake

RLBase.update!(app::NeuralNetworkApproximator, gs) = Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)
Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) = Flux.loadparams!(dest.model, Flux.params(src))

env = CarEnv(
    max_step = 500,
    control_low = [-0.2, -0.2], 
    control_high = [0.2, 0.2]
    )

reset!(env)

s_size = length(state(env))
a_size = length(action_space(env))
actor = NeuralNetworkApproximator(
    Chain(
        Dense(s_size, 64, relu),
        Dense(64, 64, relu),
        Dense(64, a_size, tanh)),
    Adam())

critic = NeuralNetworkApproximator(
    Chain(
        Dense(s_size + a_size, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 1)),
    Adam())

random_policy = RandomPolicy(Space([ClosedInterval(-1.0, 1.0), ClosedInterval(-1.0, 1.0)]))

ddpg = DDPGPolicy(
    behavior_actor = actor, 
    behavior_critic = critic, 
    target_actor = deepcopy(actor), 
    target_critic = deepcopy(critic),
    na = a_size,
    start_policy = random_policy,
    start_steps = 2000
    )

traj = CircularArraySARTTrajectory(
    capacity = 10000,
    state = Vector{Float64} => size(state(env)),
    action = Vector{Float64} => size(action_space(env))
    )

agent = Agent(ddpg, traj)
run(agent, env, StopAfterEpisode(50), TotalRewardPerEpisode())

reset!(env)
while !is_terminated(env)
    @time env(agent.policy.behavior_actor(state(env)), store = true)
    display(reward(env))
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)