using BSON
using ReinforcementLearning
using Dojo
using CairoMakie
using CarBrake

env = CarEnv(
    max_step = 1000,
    goal = [0.0, 3.0])

ppo = BSON.load("results/entropy_coef=0.01/ppo.bson")[:ppo]


reset!(env)

# X = collect(-7.0:0.1:7.0) ./ 10.0
X = collect(-7.0:0.1:3.0) ./ 10.0

Y = Float64[]
for x in X
    s = state(env)
    s[2] = x
    push!(Y, ppo.critic(s)[1])
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, X, Y)
save("value.png", fig)



# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "Space (m)", ylabel = "Space (m)")
# xlims!(ax, -8.0, 8.0)
# ylims!(ax, -8.0, 8.0)

# for i in 1:50
#     reset!(env)
#     while !is_terminated(env)
#         @time env(ppo(env), store = true)
#     end

#     x = hcat(Vector.(env.storage.x[1])...)
#     lines!(ax, x[1, :], x[2, :])
# end
# scatter!(ax, [0.0], [3.0], color = :green, marker = :rect, markersize = 40)

# save("trajectory.png", fig)





# vis = Visualizer()
# open(vis)
# visualize(env.car, env.storage, vis = vis)
