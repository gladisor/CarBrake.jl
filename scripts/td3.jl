RLBase.update!(app::NeuralNetworkApproximator, gs) = Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)
Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) = Flux.loadparams!(dest.model, Flux.params(src))

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