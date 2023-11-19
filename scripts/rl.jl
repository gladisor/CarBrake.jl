using Dojo
using ReinforcementLearning
using Flux
using IntervalSets
using CairoMakie
using CarBrake

RLBase.update!(app::NeuralNetworkApproximator, gs) = Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)
Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) = Flux.loadparams!(dest.model, Flux.params(src))


function RLBase.reward(env::CarEnv)
    body = get_body(env.car, :body)
    # return - Flux.norm(body.state.x2[1:2] .- env.goal) / 100.0
    return Flux.norm(body.state.x1[1:2] .- env.goal) .- Flux.norm(body.state.x2[1:2] .- env.goal)
end

function (p::TD3Policy)(env)
    p.update_step += 1

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = p.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp.(action .+ randn(p.rng) * p.act_noise, -p.act_limit, p.act_limit)
    end
end

function RLBase.update!(p::TD3Policy, batch::NamedTuple{SARTS})
    to_device(x) = send_to_device(device(p.behavior_actor), x)
    s, a, r, t, s′ = to_device(batch)

    actor = p.behavior_actor
    critic = p.behavior_critic

    # !!! we have several assumptions here, need revisit when we have more complex environments
    # state is vector
    # action is scalar
    target_noise =
        clamp.(
            randn(p.rng, Float32, size(a,1), p.batch_size) .* p.target_act_noise,
            -p.target_act_limit,
            p.target_act_limit,
        ) |> to_device
    # add noise and clip to act_limit bounds
    a′ = clamp.(p.target_actor(s′) + target_noise, -p.act_limit, p.act_limit)

    q_1′, q_2′ = p.target_critic(s′, a′)
    y = r .+ p.γ .* (1 .- t) .* (min.(q_1′, q_2′) |> vec)

    # ad-hoc fix to https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues/624
    if ndims(a) == 1
        a = Flux.unsqueeze(a, 1)
    end

    gs1 = gradient(Flux.params(critic)) do
        q1, q2 = critic(s, a)
        loss = Flux.mse(q1 |> vec, y) + Flux.mse(q2 |> vec, y)
        Flux.ignore() do
            p.critic_loss = loss
        end
        loss
    end
    update!(critic, gs1)

    if p.replay_counter % p.policy_freq == 0
        gs2 = gradient(Flux.params(actor)) do
            actions = actor(s)
            loss = -Flux.mean(critic.model.critic_1(vcat(s, actions)))
            Flux.ignore() do
                p.actor_loss = loss
            end
            loss
        end
        update!(actor, gs2)
        # polyak averaging
        for (dest, src) in zip(
            Flux.params([p.target_actor, p.target_critic]),
            Flux.params([actor, critic]),
        )
            dest .= p.ρ .* dest .+ (1 - p.ρ) .* src
        end
        p.replay_counter = 1
    end
    p.replay_counter += 1
end

STEPS = 1000
EPISODES = 200

## u = [forward power, steering power]
env = CarEnv(
    goal = [-5.0, 7.0],
    max_step = STEPS,
    control_low = [-0.2, -0.1], 
    control_high = [0.2, 0.1],
    axel_spring = 0.0,
    axel_damper = 2.0)

s_size = length(state(env))
a_size = length(action_space(env))
h_size = 128

# actor = NeuralNetworkApproximator(
#     Chain(
#         Dense(s_size, h_size, relu),
#         Dense(h_size, h_size, relu),
#         Dense(h_size, a_size, tanh)),
#     Adam()
#     )

# build_critic() = Chain(
#     Dense(s_size + a_size, h_size, relu),
#     Dense(h_size, h_size, relu),
#     Dense(h_size, h_size, relu),
#     Dense(h_size, h_size, relu),
#     Dense(h_size, 1))

# # td3_critic = TD3Critic(build_critic(), build_critic())
# # critic = NeuralNetworkApproximator(td3_critic, Adam())
# critic = NeuralNetworkApproximator(build_critic(), Adam())

# random_policy = RandomPolicy(Space([ClosedInterval(-1.0, 1.0), ClosedInterval(-1.0, 1.0)]))

# ddpg = DDPGPolicy(
#     behavior_actor = actor, 
#     behavior_critic = critic, 
#     target_actor = deepcopy(actor), 
#     target_critic = deepcopy(critic),
#     start_policy = random_policy,
#     act_noise = 1.0,
#     na = a_size,
#     γ = 0.80f0,
#     )

traj = CircularArraySARTTrajectory(
    capacity = STEPS * EPISODES,
    state = Vector{Float64} => size(state(env)),
    action = Vector{Float64} => size(action_space(env)))

agent = Agent(ddpg, traj)
# hook = TotalRewardPerEpisode()
run(agent, env, StopAfterEpisode(EPISODES), hook)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, hook.rewards)
save("ddpg_rewards_relative_rf.png", fig)

reset!(env)
while !is_terminated(env)
    @time env(agent.policy.behavior_actor(state(env)), store = true)
    display(reward(env))
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)

