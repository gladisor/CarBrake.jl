export get_car, initialize_car!, get_control_to_input

WHEEL_COLOR = RGBA(76/255, 184/255, 224/255, 1.0)
BODY_COLOR = RGBA(76/255, 224/255, 141/255, 1.0)

function get_car(;
        μ::Float64 = 0.8, 
        # axel_spring::Float64 = 1.0, 
        # axel_damper::Float64 = 2.0
        axel_spring::Float64 = 0.0, 
        axel_damper::Float64 = 0.0
        )
    origin = Origin()

    body_x = 1.5
    body_y = 2.0
    body_z = 0.25
    body_m = body_x*body_y*body_z
    body = Dojo.Box(body_x, body_y, body_z, body_m, name = :body, color = BODY_COLOR)

    wheel_r = 0.3
    wheel_h = 0.1
    wheel_m = pi * wheel_r^2 * wheel_h

    front_axel = Dojo.Sphere(0.10, 0.10, name = :front_axel)
    front_wheel = Dojo.Cylinder(0.10, wheel_h, 0.10, name = :front_wheel, orientation_offset = Dojo.RotY(pi/2), color = WHEEL_COLOR)

    fl_wheel = Dojo.Cylinder(wheel_r, wheel_h, wheel_m, name = :fl_wheel, orientation_offset = Dojo.RotY(pi/2), color = WHEEL_COLOR)
    fr_wheel = Dojo.Cylinder(wheel_r, wheel_h, wheel_m, name = :fr_wheel, orientation_offset = Dojo.RotY(pi/2), color = WHEEL_COLOR)

    bl_wheel = Dojo.Cylinder(wheel_r, wheel_h, wheel_m, name = :bl_wheel, orientation_offset = Dojo.RotY(pi/2), color = WHEEL_COLOR)
    br_wheel = Dojo.Cylinder(wheel_r, wheel_h, wheel_m, name = :br_wheel, orientation_offset = Dojo.RotY(pi/2), color = WHEEL_COLOR)

    bodies = [
        body, 
        front_axel,
        front_wheel,

        fl_wheel, fr_wheel,
        bl_wheel, br_wheel
        ]

    body_joint = JointConstraint(Dojo.Floating(origin, body), name = :body_joint)

    wheel_x_offset = Dojo.X_AXIS * body_x / 2.0
    wheel_y_offset = Dojo.Y_AXIS * body_y / 2.0
    wheel_z_offset = Dojo.Z_AXIS * body_z / 2.0

    θ = 0.60 ## max turning angle?
    # θ = 0.45
    front_axel_joint = JointConstraint(Dojo.Revolute(
        body, front_axel, Dojo.Z_AXIS,
        parent_vertex = wheel_y_offset .- wheel_z_offset,
        rot_joint_limits = [-θ*sones(1), θ*sones(1)],
        spring = axel_spring,
        damper = axel_damper),
        name = :front_axel_joint)

    front_wheel_joint = JointConstraint(Dojo.Revolute(front_axel, front_wheel, Dojo.X_AXIS))

    fl_wheel_joint = JointConstraint(Dojo.Spherical(
        body, fl_wheel,
        parent_vertex = wheel_x_offset .+ wheel_y_offset .- wheel_z_offset
        ))

    fl_wheel_lock = JointConstraint(Dojo.FixedOrientation(front_wheel, fl_wheel))

    fr_wheel_joint = JointConstraint(Dojo.Spherical(
        body, fr_wheel,
        parent_vertex = -wheel_x_offset .+ wheel_y_offset .- wheel_z_offset
        ))

    fr_wheel_lock = JointConstraint(Dojo.FixedOrientation(front_wheel, fr_wheel))

    bl_wheel_joint = JointConstraint(Dojo.Revolute(
        body, bl_wheel, Dojo.X_AXIS, 
        parent_vertex = wheel_x_offset .- wheel_y_offset .- wheel_z_offset
        ))

    br_wheel_joint = JointConstraint(Dojo.Revolute(
        body, br_wheel, Dojo.X_AXIS, 
        parent_vertex = -wheel_x_offset .- wheel_y_offset .- wheel_z_offset
        ))

    joints = [
        body_joint,         ## (Floating) 6
        front_axel_joint,   ## (Revolute) 1
        front_wheel_joint,  ## (Revolute) 1

        fl_wheel_joint,     ## (Spherical) 3
        fl_wheel_lock,      ## (FixedOrientation) 3
        
        fr_wheel_joint,     ## (Spherical) 3
        fr_wheel_lock,      ## (FixedOrientation) 3

        bl_wheel_joint,     ## (Revolute) 1
        br_wheel_joint      ## (Revolute) 1
        ] ## 22 dof

    contacts = [
        contact_constraint(fl_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(fr_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(bl_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(br_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ)
        ]

    return Mechanism(origin, bodies, joints, contacts)
end

function add_goal(car::Mechanism, goal::Vector)
    x, y, z = 1.0, 1.0, 2.0
    goal_box = Dojo.Box(x, y, z, x*y*z, name = :goal_box, color = RGBA(0.0, 1.0, 0.0, 0.33))
    bodies = [car.bodies; goal_box]
    goal_joint = JointConstraint(Fixed(car.origin, goal_box, parent_vertex = [goal; z/2.0]), name = :goal_joint)
    joints = [car.joints; goal_joint]
    return Mechanism(car.origin, bodies, joints, car.contacts)
end

function initialize_car!(mech::Mechanism, x_car::Float64, y_car::Float64)
        # , x_goal::Float64, y_goal::Float64)
    zero_coordinates!(mech)
    zero_velocities!(mech)

    car_pos = Dojo.X_AXIS * x_car .+ Dojo.Y_AXIS * y_car .+ Dojo.Z_AXIS * 0.5
    set_minimal_coordinates!(mech, get_joint(mech, :body_joint), vcat(car_pos, zeros(3)))

    # goal_box = get_body(mech, :goal_box)
    # goal_pos = Dojo.X_AXIS * x_goal .+ Dojo.Y_AXIS * y_goal .+ Dojo.Z_AXIS * goal_box.shape.xyz[3]
    # set_minimal_coordinates!(mech, get_joint(mech, :goal_joint), vcat(goal_pos, zeros(3)))
end

function get_torque_mask(car::Mechanism)
    u = zeros(input_dimension(car))
    u[[end-1, end]] .= 1
    return u
end

function get_steering_mask(car::Mechanism)
    u = zeros(input_dimension(car))
    u[7] = 1
    return u
end

function get_control_to_input(car::Mechanism)
    τ = get_torque_mask(car)
    θ = get_steering_mask(car)
    return hcat(τ, θ)
end