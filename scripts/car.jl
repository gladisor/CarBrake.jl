using Dojo
using CarBrake

mech = get_car()
initialize_car!(mech)

actions = car_actions()

K = 500
storage = Storage(K, length(mech.bodies))

for k in 1:K
    # u = actions[:, rand(axes(actions, 2))]
    @time step!(mech, get_maximal_state(mech), CarBrake.car_turn_right())
    Dojo.save_to_storage!(mech, storage, k)
end

vis = Visualizer()
open(vis)
visualize(mech, storage, vis = vis)