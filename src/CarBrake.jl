module CarBrake

using Dojo
using StaticArrays
using ReinforcementLearning

const FRICTION_PARAMETERIZATION = SA{Float64}[
    1.0  0.0
    0.0  1.0]

include("mechanisms/car.jl")

end
