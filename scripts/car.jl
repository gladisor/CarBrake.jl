using Dojo
using CarBrake
include("../src/mechanisms/car.jl")

mech = get_car()
initialize_car!(mech)

actions = car_actions()

K = 500
storage = Storage(K, length(mech.bodies))

for k in 1:K
    Jx, Ju = get_minimal_gradients!(mech, get_minimal_state(mech), actions[:, rand(axes(actions, 2))])
    # Jx, Ju = get_minimal_gradients!(mech, get_minimal_state(mech), car_forward())
    Dojo.save_to_storage!(mech, storage, k)
end

vis = Visualizer()
open(vis)
visualize(mech, storage, vis = vis)