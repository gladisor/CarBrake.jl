using ReinforcementLearning
using Flux
using Flux: Optimiser
using Distributions

struct Value
    layers::Chain
end

Flux.@functor Value

function Value(s_size::Int, h_size::Int)
    layers = Chain(
        Dense(s_size, h_size, tanh),
        Dense(h_size, h_size, tanh),
        Dense(h_size, 1))

    return Value(layers)
end

function (value::Value)(s)
    return value.layers(s)
end

struct Actor
    pre::Chain
    μ::Dense
    logσ::Dense
    a_lim::Real
end

Flux.@functor Actor
Flux.trainable(actor::Actor) = (actor.pre, actor.μ, actor.logσ)

function Actor(s_size::Int, h_size::Int, a_size::Int; a_lim::Real = 1.0)
    pre = Chain(
        Dense(s_size, h_size, tanh),
        Dense(h_size, h_size, tanh)
        )

    μ = Dense(h_size, a_size)
    logσ = Dense(h_size, a_size)

    return Actor(pre, μ, logσ, a_lim)
end

function (actor::Actor)(s)
    x = actor.pre(s)
    μ = actor.μ(x)
    σ = exp.(clamp.(actor.logσ(x), -20, 10))
    ϵ = randn(Float32, size(σ))
    a = tanh.(μ .+ σ .* ϵ) * actor.a_lim
    return a, μ, σ
end

mutable struct PPO <: AbstractPolicy
    actor
    critic
    target_actor
    target_critic

    actor_opt
    critic_opt

    actor_loss::Float32
    critic_loss::Float32

    γ::Float32
    ρ::Float32
end

function PPO(;actor, critic, actor_opt, critic_opt, γ, ρ)
    return PPO(
        actor, critic, 
        deepcopy(actor), deepcopy(critic), 
        actor_opt, critic_opt, 
        0.0f0, 0.0f0, γ, ρ)
end

function (ppo::PPO)(env::CarEnv)
    a, μ, σ = ppo.actor(state(env))
    return a
end

function build_buffer(env, episodes::Int)

    episode_length = env.max_step

    traj = CircularArraySARTTrajectory(
            capacity = episodes * episode_length,
            state = Vector{Float64} => size(state(env)),
            action = Vector{Float64} => size(action_space(env)))

    return traj
end

function soft_update!(target, source, ρ)
    for (targ, src) in zip(Flux.params(target), Flux.params(source))
        targ .= ρ .* targ .+ (1 - ρ) .* src
    end
end

function sarts(traj::CircularArraySARTTrajectory)
    idx = 1:length(traj)
    s, a, r, t = (convert(Array, select_last_dim(traj[x], idx)) for x in SART)
    s′ = convert(Array, select_last_dim(traj[:state], idx .+ 1))
    return (s, a, r, t, s′)
end

using Dojo
Flux.device!(2)
using CairoMakie
using CarBrake
include("rl.jl")

env = CarEnv(
    max_step = 1000,
    goal = [0.0, 3.0],
    control_low = [-0.3, -1.0],
    control_high = [0.3, 1.0])

s_size = length(state(env))
a_size = length(action_space(env))
h_size = 128

# ppo = PPO(
#     actor = Actor(s_size, 64, a_size, a_lim = 1.0f0),
#     critic = Value(s_size, 128),
#     actor_opt = Optimiser(ClipNorm(1f-4), Adam(0.00005f0)),
#     critic_opt = Optimiser(ClipNorm(1f-4), Adam(0.0003f0)),
#     γ = 0.95f0,
#     ρ = 0.995f0)

γ = ppo.γ
episodes = 20
epochs = 5
actor_ps = Flux.params(ppo.actor)
critic_ps = Flux.params(ppo.critic)

hook = TotalRewardPerEpisode()
actor_loss = Float64[]
critic_loss = Float64[]

# agent = Agent(policy = ppo, trajectory = build_buffer(env, episodes))
# run(agent, env, StopAfterEpisode(episodes), hook)
# data = sarts(agent.trajectory)
# batches = Flux.DataLoader(data, batchsize = 32, shuffle = true)
# batch = first(batches)
# s, a, r, t, s′ = batch

# critic_gs = Flux.gradient(critic_ps) do 
#     y = r .+ γ * vec(ppo.target_critic(s′))
#     δ = y .- vec(ppo.critic(s))
#     loss = mean(δ .^ 2)

#     Flux.ignore() do 
#         ppo.critic_loss = loss
#     end
#     return loss
# end

# _, μ_old, σ_old = ppo.target_actor(s)
# old_p_a = prod(pdf.(Normal.(μ_old, σ_old), a), dims = 1) |> vec


for iteration ∈ 1:100

    agent = Agent(policy = ppo, trajectory = build_buffer(env, episodes))
    run(agent, env, StopAfterEpisode(episodes), hook)

    data = sarts(agent.trajectory)
    batches = Flux.DataLoader(data, batchsize = 32, shuffle = true)

    for epoch ∈ 1:epochs

        for batch ∈ batches

            s, a, r, t, s′ = batch
            critic_gs = Flux.gradient(critic_ps) do 
                y = r .+ γ * vec(ppo.target_critic(s′))
                δ = y .- vec(ppo.critic(s))
                loss = mean(δ .^ 2)

                Flux.ignore() do 
                    ppo.critic_loss = loss
                end
                return loss
            end

            Flux.update!(ppo.critic_opt, critic_ps, critic_gs)

            actor_gs = Flux.gradient(actor_ps) do
                _, μ_old, σ_old = ppo.target_actor(s)
                old_p_a = prod(pdf.(Normal.(μ_old, σ_old), a), dims = 1) |> vec

                _, μ, σ = ppo.actor(s)
                p_a = prod(pdf.(Normal.(μ, σ), a), dims = 1) |> vec
                ratio = p_a ./ old_p_a

                δ = r .+ γ * vec(ppo.target_critic(s′)) .- vec(ppo.critic(s))

                δ = (δ .- mean(δ)) ./ std(δ)

                loss = -mean(min.(ratio .* δ, clamp.(ratio, 0.9f0, 1.1f0) .* δ)) + mean((1 .- ratio) .^ 2) + mean(ratio .* log.(p_a)) * 0.001

                Flux.ignore() do 
                    ppo.actor_loss = loss
                end

                return loss
            end

            Flux.update!(ppo.actor_opt, actor_ps, actor_gs)

            soft_update!(ppo.target_critic, ppo.critic, ppo.ρ)
            push!(actor_loss, ppo.actor_loss)
            push!(critic_loss, ppo.critic_loss)
        end
    end

    soft_update!(ppo.target_actor, ppo.actor, 0.0f0)

    fig = Figure()
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    ax3 = Axis(fig[1, 3])
    lines!(ax1, actor_loss)
    lines!(ax2, critic_loss)
    lines!(ax3, hook.rewards)
    save("ppo.png", fig)
end