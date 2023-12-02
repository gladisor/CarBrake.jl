module CarBrake

using Dojo
using StaticArrays
using ReinforcementLearning
using IntervalSets
using Flux
using Distributions
using BSON
using CairoMakie

const FRICTION_PARAMETERIZATION = SA{Float64}[
    1.0  0.0
    0.0  1.0]

include("mechanisms/car.jl")
include("environments/car.jl")
include("ppo.jl")

end
