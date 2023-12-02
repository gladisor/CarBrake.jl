using Flux
using ReinforcementLearning
using Dojo
using CarBrake

env = CarEnv(
    max_step = 1000,
    goal = [0.0, 3.0])

s_size = length(state(env))
a_size = length(action_space(env))
h_size = 128

ppo = PPO(
    actor = Actor(s_size, h_size, a_size),
    critic = Value(s_size, h_size),
    actor_opt = Adam(0.00005f0),
    critic_opt = Adam(0.0003f0),
    γ = 0.99f0,
    ρ = 0.995f0)

# ppo = BSON.load("ppo.bson")[:ppo]

ppo = train!(
    ppo, 
    env, 
    entropy_coef = 0.01f0,
    path = "results/entropy_coef=0.01")

reset!(env)
while !is_terminated(env)
    @time env(ppo(env), store = true)
end

vis = Visualizer()
open(vis)
visualize(env.car, env.storage, vis = vis)
