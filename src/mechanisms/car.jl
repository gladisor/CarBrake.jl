export get_car, initialize_car!, car_actions

WHEEL_COLOR = RGBA(76/255, 184/255, 224/255, 1.0)
BODY_COLOR = RGBA(76/255, 224/255, 141/255, 1.0)

function get_car(;μ::Float64)
    origin = Origin()

    body_x = 1.5
    body_y = 2.0
    body_z = 0.25
    body_m = body_x * body_y * body_z
    body = Dojo.Box(body_x, body_y, body_z, body_m, name = :body, color = BODY_COLOR)

    wheel_r = 0.3
    wheel_h = 0.1
    wheel_m = pi * wheel_r^2 * wheel_h

    front_axel = Dojo.Sphere(0.10, 0.10, name = :front_axel)
    front_wheel = Dojo.Cylinder(wheel_r, wheel_h, 0.10, name = :bl_wheel, orientation_offset = Dojo.RotY(pi/2), color = WHEEL_COLOR)

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

    θ = 0.60
    front_axel_joint = JointConstraint(Dojo.Revolute(
        body, front_axel, Dojo.Z_AXIS,
        parent_vertex = wheel_y_offset .- wheel_z_offset,
        rot_joint_limits = [-θ*sones(1), θ*sones(1)]
    ))

    front_wheel_joint = JointConstraint(Dojo.Revolute(
        front_axel, front_wheel, Dojo.X_AXIS,
        ))

    fl_wheel_joint = JointConstraint(Dojo.Spherical(
        body, fl_wheel,
        parent_vertex = wheel_x_offset .+ wheel_y_offset .- wheel_z_offset
    ))

    fl_wheel_lock = JointConstraint(Dojo.FixedOrientation(
        front_wheel, fl_wheel
    ))

    fr_wheel_joint = JointConstraint(Dojo.Spherical(
        body, fr_wheel,
        parent_vertex = -wheel_x_offset .+ wheel_y_offset .- wheel_z_offset
    ))

    fr_wheel_lock = JointConstraint(Dojo.FixedOrientation(
        front_wheel, fr_wheel
    ))

    bl_wheel_joint = JointConstraint(Dojo.Revolute(
        body, bl_wheel, Dojo.X_AXIS, 
        parent_vertex = wheel_x_offset .- wheel_y_offset .- wheel_z_offset
        ))

    br_wheel_joint = JointConstraint(Dojo.Revolute(
        body, br_wheel, Dojo.X_AXIS, 
        parent_vertex = -wheel_x_offset .- wheel_y_offset .- wheel_z_offset
        ))

    joints = [
        body_joint,         ## 6 dof (Floating)
        front_axel_joint,   ## 1 dof (Revolute)
        front_wheel_joint,  ## 1 dof (Revolute)

        fl_wheel_joint,     ## 3 dof (Spherical)
        fl_wheel_lock,      ## 0 dof (FixedOrientation)
        
        fr_wheel_joint,     ## 3 dof (Spherical)
        fr_wheel_lock,      ## 0 dof (FixedOrientation)

        bl_wheel_joint,     ## 1 dof (Revolute)
        br_wheel_joint      ## 1 dof (Revolute)
        ]

    contacts = [
        # contact_constraint(front_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(fl_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(fr_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(bl_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ),
        contact_constraint(br_wheel, Dojo.Z_AXIS, contact_radius = wheel_r, friction_coefficient = μ)
        ]

    return Mechanism(origin, bodies, joints, contacts)
end

function initialize_car!(mech::Mechanism, x::Float64, y::Float64)
    zero_coordinates!(mech)
    zero_velocities!(mech)
    X = Dojo.X_AXIS * x .+ Dojo.Y_AXIS * y .+ Dojo.Z_AXIS * 1.0
    set_minimal_coordinates!(mech, get_joint(mech, :body_joint), vcat(X, zeros(3)))
end

car_forward() = vcat(zeros(Float64, 6), [1.0, 1.0, 1.0, 1.0])
car_backward() = -car_forward()
car_turn_right() = vcat(zeros(Float64, 6), [2.0, -2.0, 2.0, -2.0])
car_turn_left() = -car_turn_right()
car_idle() = zeros(Float64, 10)
car_actions() = hcat(car_forward(), car_backward(), car_turn_right(), car_turn_left(), car_idle())