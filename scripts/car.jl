using Dojo
using CarBrake

function add_brakes(mech::Mechanism; disc_mass::Float64 = 1.0)

    wheel = get_body(mech, :bl_wheel)
    r = wheel.shape.rh[1]
    h = wheel.shape.rh[2]

    bl_disc = Dojo.Box(h + 0.05, r*sqrt(2), r*sqrt(2), disc_mass, name = :bl_disc)
    br_disc = Dojo.Box(h + 0.05, r*sqrt(2), r*sqrt(2), disc_mass, name = :br_disc)
    bodies = [mech.bodies; bl_disc; br_disc]

    bl_disc_joint = JointConstraint(Fixed(get_body(mech, :bl_wheel), bl_disc))
    br_disc_joint = JointConstraint(Fixed(get_body(mech, :br_wheel), br_disc))
    joints = [mech.joints; bl_disc_joint; br_disc_joint]
    return Mechanism(mech.origin, bodies, joints, mech.contacts)
end

car = get_car()
car = add_brakes(car)
initialize_car!(car)

actions = car_actions()

K = 500
storage = Storage(K, length(car.bodies))

for k in 1:K
    u = actions[:, rand(axes(actions, 2))]
    @time step!(car, get_maximal_state(car), u)
    Dojo.save_to_storage!(car, storage, k)
end

vis = Visualizer()
open(vis)
visualize(car, storage, vis = vis)