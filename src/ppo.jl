export Value, Actor, PPO, train!

struct Value
    layers::Chain
end

Flux.@functor Value

function Value(s_size::Int, h_size::Int)
    layers = Chain(
        Dense(s_size, h_size, relu),
        Dense(h_size, h_size, relu),
        Dense(h_size, h_size, relu),
        Dense(h_size, h_size, relu),
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

function Actor(s_size::Int, h_size::Int, a_size::Int; a_lim::Real = 1.0f0)
    pre = Chain(
        Dense(s_size, h_size, relu),
        Dense(h_size, h_size, relu),
        Dense(h_size, h_size, relu)
        )

    μ = Dense(h_size, a_size)
    logσ = Dense(h_size, a_size)

    return Actor(pre, μ, logσ, a_lim)
end

function (actor::Actor)(s)
    x = actor.pre(s)
    μ = actor.μ(x)
    σ = exp.(clamp.(actor.logσ(x), -20.0, 10.0))
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

    pg_loss::Real
    entropy_loss::Real
    critic_loss::Real

    γ::Float32
    ρ::Float32
end

function PPO(;actor, critic, actor_opt, critic_opt, γ, ρ)
    return PPO(
        actor, critic, 
        deepcopy(actor), deepcopy(critic), 
        actor_opt, critic_opt, 
        0.0f0, 0.0f0, 0.0f0, γ, ρ)
end

function (ppo::PPO)(env::CarEnv)
    a, _, _ = ppo.actor(state(env))
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

function train!(ppo::PPO, env::CarEnv; 
        iterations = 100,
        episodes::Int = 20,
        epochs::Int = 5,
        entropy_coef::Float32 = 0.01f0,
        path::String,
    )

    # actor_ps = Flux.params(ppo.actor)
    # critic_ps = Flux.params(ppo.critic)
    hook = TotalRewardPerEpisode()

    γ = ppo.γ
    ϵ = 0.1f0

    pg_loss_history = Float64[]
    entropy_loss_history = Float64[]
    critic_loss_history = Float64[]

    for _ ∈ 1:iterations

        agent = Agent(policy = ppo, trajectory = build_buffer(env, episodes))
        run(agent, env, StopAfterEpisode(episodes), hook)

        data = sarts(agent.trajectory)
        batches = Flux.DataLoader(data, batchsize = 64, shuffle = true, partial = false)

        actor_ps = Flux.params(ppo.actor)
        critic_ps = Flux.params(ppo.critic)

        for _ ∈ 1:epochs
            for batch ∈ batches

                s, a, r, t, s′ = batch
                critic_gs = Flux.gradient(critic_ps) do 
                    y = r .+ γ * vec(ppo.target_critic(s′)) .* .!t
                    δ = y .- vec(ppo.critic(s))
                    loss = Flux.mean(δ .^ 2)

                    Flux.ignore() do 
                        ppo.critic_loss = Float64(loss)
                    end
                    return loss
                end

                actor_gs = Flux.gradient(actor_ps) do
                    _, μ_old, σ_old = ppo.target_actor(s)
                    old_p_a = prod(pdf.(Normal.(μ_old, σ_old), a), dims = 1) |> vec

                    _, μ, σ = ppo.actor(s)
                    p_a = prod(pdf.(Normal.(μ, σ), a), dims = 1) |> vec
                    p_a = ifelse.(isnan.(p_a), 0.0f0, p_a)
                    ratio = (p_a .+ 1f-10) ./ (old_p_a .+ 1f-10)

                    y = r .+ γ * vec(ppo.target_critic(s′)) .* .!t
                    δ = y .- vec(ppo.critic(s))
                    # δ = r .+ γ * vec(ppo.target_critic(s′)) .- vec(ppo.critic(s))
                    δ = (δ .- Flux.mean(δ)) ./ (Flux.std(δ) .+ 1f-10)

                    pg_loss = -Flux.mean(min.(ratio .* δ, clamp.(ratio, 1.0f0 - ϵ, 1.0f0 + ϵ) .* δ))
                    entropy_loss = -Flux.mean(p_a .* log.(p_a .+ 1f-10))
                    loss = pg_loss - entropy_loss * entropy_coef

                    Flux.ignore() do 
                        # ppo.actor_loss = loss
                        ppo.pg_loss = Float64(pg_loss)
                        ppo.entropy_loss = Float64(entropy_loss)
                    end
                    return loss
                end

                Flux.update!(ppo.critic_opt, critic_ps, critic_gs)
                Flux.update!(ppo.actor_opt, actor_ps, actor_gs)

                soft_update!(ppo.target_critic, ppo.critic, ppo.ρ)
                # push!(actor_loss, ppo.actor_loss)

                push!(pg_loss_history, ppo.pg_loss)
                push!(entropy_loss_history, ppo.entropy_loss)
                push!(critic_loss_history, ppo.critic_loss)
            end
        end

        soft_update!(ppo.target_actor, ppo.actor, 0.0f0)

        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Episode", ylabel = "Episode Reward")
        lines!(ax, hook.rewards)
        save(joinpath(path, "reward.png"), fig)

        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Update Step", ylabel = "Policy Gradient Loss")
        lines!(ax, pg_loss_history)
        save(joinpath(path, "policy_gradient_loss.png"), fig)

        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Update Step", ylabel = "Entropy Loss")
        lines!(ax, entropy_loss_history)
        save(joinpath(path, "entropy_loss.png"), fig)

        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Update Step", ylabel = "Critic Loss")
        lines!(ax, critic_loss_history)
        save(joinpath(path, "critic_loss.png"), fig)

        bson(joinpath(path, "ppo.bson"), ppo = ppo)
    end

    return ppo
end