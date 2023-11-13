using Dojo, StaticArrays

function get_car()

    μ = 0.1
    origin = Origin()

    body_x = 1.5
    body_y = 2.0
    body_z = 0.5
    body = Dojo.Box(body_x, body_y, body_z, 1.0, name = :body)

    wheel_r = 0.3
    wheel_m = 1.0
    fl_wheel = Dojo.Sphere(wheel_r, wheel_m, name = :fl_wheel)
    fr_wheel = Dojo.Sphere(wheel_r, wheel_m, name = :fr_wheel)
    bl_wheel = Dojo.Sphere(wheel_r, wheel_m, name = :bl_wheel)
    br_wheel = Dojo.Sphere(wheel_r, wheel_m, name = :br_wheel)

    bodies = [body, fl_wheel, fr_wheel, bl_wheel, br_wheel]

    body_joint = JointConstraint(Dojo.Floating(origin, body))

    wheel_x_offset = Dojo.X_AXIS * body_x / 2.0
    wheel_y_offset = Dojo.Y_AXIS * body_y / 2.0
    wheel_z_offset = Dojo.Z_AXIS * body_z / 2.0
    fl_wheel_joint = JointConstraint(Dojo.Revolute(
        body, fl_wheel, Dojo.X_AXIS, 
        parent_vertex = wheel_x_offset .+ wheel_y_offset .- wheel_z_offset
        ))
    fr_wheel_joint = JointConstraint(Dojo.Revolute(
        body, fr_wheel, Dojo.X_AXIS, 
        parent_vertex = -wheel_x_offset .+ wheel_y_offset .- wheel_z_offset
        ))
    bl_wheel_joint = JointConstraint(Dojo.Revolute(
        body, bl_wheel, Dojo.X_AXIS, 
        parent_vertex = wheel_x_offset .- wheel_y_offset .- wheel_z_offset
        ))
    br_wheel_joint = JointConstraint(Dojo.Revolute(
        body, br_wheel, Dojo.X_AXIS, 
        parent_vertex = -wheel_x_offset .- wheel_y_offset .- wheel_z_offset
        ))

    contacts = [
        contact_constraint(fl_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ)
        contact_constraint(fr_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ)
        contact_constraint(bl_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ)
        contact_constraint(br_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ)
        ]

    joints = [
        body_joint, 
        fl_wheel_joint, 
        fr_wheel_joint,
        bl_wheel_joint, 
        br_wheel_joint
        ]

    return Mechanism(origin, bodies, joints, contacts)
end

mech = get_car()

zero_coordinates!(mech)
zero_velocities!(mech)

x = Dojo.Z_AXIS * 3
set_maximal_configurations!(get_body(mech, :body), x = x)
set_maximal_configurations!(get_body(mech, :fl_wheel), x = x)
set_maximal_configurations!(get_body(mech, :fr_wheel), x = x)
set_maximal_configurations!(get_body(mech, :bl_wheel), x = x)
set_maximal_configurations!(get_body(mech, :br_wheel), x = x)

u = zeros(input_dimension(mech))

car_forward() = vcat(zeros(Float64, 6), [1.0, 1.0, 1.0, 1.0])
car_backward() = -car_forward()
car_turn_right() = vcat(zeros(Float64, 6), [1.0, -1.0, 1.0, -1.0])
car_turn_left() = -car_turn_right()
car_actions() = hcat(car_forward(), car_backward(), car_turn_right(), car_turn_left())
actions = car_actions()

K = 500
storage = Storage(K, length(mech.bodies))

for k in 1:K
    Jx, Ju = get_minimal_gradients!(mech, get_minimal_state(mech), actions[:, rand(1:4)])
    Dojo.save_to_storage!(mech, storage, k)
end

vis = Visualizer()
open(vis)
visualize(mech, storage, vis = vis)

# storage = simulate!(mech, 1.0)

# vis = Visualizer()
# open(vis)
# visualize(mech, storage, vis = vis)
