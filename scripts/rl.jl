using Dojo
using ReinforcementLearning
using Flux
using CairoMakie
using CarBrake
# include("td3.jl")

function RLBase.state(env::CarEnv)
    body = get_body(env.car, :body)
    front_axel = get_body(env.car, :front_axel)
    t = env.step / env.max_step

    return Vector{Float64}(vcat(
        body.state.x2[1:2] ./ 10.0,    ## only care about position in xy
        body.state.v15[1:2],   ## only care about velocity in xy
        Dojo.vector(body.state.q2),
        body.state.ω15,
        Dojo.vector(front_axel.state.q2),
        front_axel.state.ω15,    ## how fast is the steering wheel turning
        t
        ))
end