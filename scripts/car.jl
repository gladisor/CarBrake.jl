using Dojo, StaticArrays
using CarBrake
include("../src/mechanisms/car.jl")

# PAD_COLOR = RGBA(249/255, 129/255, 191/255, 1.0)

# function add_brake_discs(mech::Mechanism)
#     ## getting wheel for geometric properties
#     wheel = get_body(mech, :bl_wheel)
#     r = wheel.shape.rh[1]
#     h = wheel.shape.rh[2]
#     s = r * sqrt(2) ## maximum side length of a square inscribed in a circle
#     disc_mass = s * s * h

#     ## defining discs which are fixed to each wheel which provide a contact surface
#     fl_disc = Dojo.Box(h + 0.05, s, s, disc_mass, name = :fl_disc)
#     fr_disc = Dojo.Box(h + 0.05, s, s, disc_mass, name = :fr_disc)
#     bl_disc = Dojo.Box(h + 0.05, s, s, disc_mass, name = :bl_disc)
#     br_disc = Dojo.Box(h + 0.05, s, s, disc_mass, name = :br_disc)
#     bodies = [mech.bodies; fl_disc; fr_disc; bl_disc; br_disc]

#     ## fixing each disc to the wheel
#     fl_disc_joint = JointConstraint(Fixed(get_body(mech, :fl_wheel), fl_disc))
#     fr_disc_joint = JointConstraint(Fixed(get_body(mech, :fr_wheel), fr_disc))
#     bl_disc_joint = JointConstraint(Fixed(get_body(mech, :bl_wheel), bl_disc))
#     br_disc_joint = JointConstraint(Fixed(get_body(mech, :br_wheel), br_disc))

#     joints = [mech.joints; fl_disc_joint; fr_disc_joint; bl_disc_joint; br_disc_joint]
#     return Mechanism(mech.origin, bodies, joints, mech.contacts)
# end

# function sphere_box_contact(sphere::Dojo.Body, box::Dojo.Body, μ::Float64; name::Symbol)
#     collision = SphereBoxCollision{Float64, 2, 3, 6}(szeros(3), box.shape.xyz..., sphere.shape.r)
#     contact = NonlinearContact{Float64, 8}(μ, CarBrake.FRICTION_PARAMETERIZATION, collision)
#     return ContactConstraint((contact, sphere.id, box.id); name)
# end

# function add_brake_pads(mech::Mechanism; pad_r::Float64=0.05, pad_m::Float64=1.0, μ::Float64=0.5, pad_spring::Float64=10.0, pad_damper::Float64=1.0)
#     fl_pad = Dojo.Sphere(pad_r, pad_m, name = :fl_pad, color = PAD_COLOR)
#     fr_pad = Dojo.Sphere(pad_r, pad_m, name = :fr_pad, color = PAD_COLOR)
#     bl_pad = Dojo.Sphere(pad_r, pad_m, name = :bl_pad, color = PAD_COLOR)
#     br_pad = Dojo.Sphere(pad_r, pad_m, name = :br_pad, color = PAD_COLOR)
#     bodies = [mech.bodies; fl_pad; fr_pad; bl_pad; br_pad]

#     ## obtaining main car body to get geometric properties
#     body = get_body(mech, :body)
#     disc = get_body(mech, :fl_disc)
#     s = disc.shape.xyz[2]

#     ## defining offsets for brake pads
#     left_x_offset = (body.shape.xyz[1] / 2.0 + pad_r + disc.shape.xyz[1])
#     right_x_offset = -left_x_offset
#     front_y_offset = (body.shape.xyz[2] / 2.0)
#     back_y_offset = -front_y_offset
#     z_offset = (body.shape.xyz[3]/2.0 + s/2.0 - pad_r)

#     ## defining positions of each pad
#     fl_pad_offset = Dojo.X_AXIS * left_x_offset  .+ Dojo.Y_AXIS * front_y_offset .- Dojo.Z_AXIS * z_offset
#     fr_pad_offset = Dojo.X_AXIS * right_x_offset .+ Dojo.Y_AXIS * front_y_offset .- Dojo.Z_AXIS * z_offset
#     bl_pad_offset = Dojo.X_AXIS * left_x_offset  .+ Dojo.Y_AXIS * back_y_offset  .- Dojo.Z_AXIS * z_offset
#     br_pad_offset = Dojo.X_AXIS * right_x_offset .+ Dojo.Y_AXIS * back_y_offset  .- Dojo.Z_AXIS * z_offset

#     ## creating prismatic joints for 1 translational degree of freedom
#     fl_joint = JointConstraint(Dojo.Prismatic(body, fl_pad, Dojo.X_AXIS, parent_vertex = fl_pad_offset, spring = pad_spring, damper = pad_damper))
#     fr_joint = JointConstraint(Dojo.Prismatic(body, fr_pad, Dojo.X_AXIS, parent_vertex = fr_pad_offset, spring = pad_spring, damper = pad_damper))
#     bl_joint = JointConstraint(Dojo.Prismatic(body, bl_pad, Dojo.X_AXIS, parent_vertex = bl_pad_offset, spring = pad_spring, damper = pad_damper))
#     br_joint = JointConstraint(Dojo.Prismatic(body, br_pad, Dojo.X_AXIS, parent_vertex = br_pad_offset, spring = pad_spring, damper = pad_damper))
#     joints = [mech.joints; fl_joint; fr_joint; bl_joint; br_joint]

#     contacts = [
#         mech.contacts; 
#         sphere_box_contact(fl_pad, get_body(mech, :fl_disc), μ, name = :fl_pad_contact);
#         sphere_box_contact(fr_pad, get_body(mech, :fr_disc), μ, name = :fr_pad_contact);
#         sphere_box_contact(bl_pad, get_body(mech, :bl_disc), μ, name = :bl_pad_contact);
#         sphere_box_contact(br_pad, get_body(mech, :br_disc), μ, name = :br_pad_contact)
#         ]
    
#     return Mechanism(mech.origin, bodies, joints, contacts)
# end

function torque_mask(car::Mechanism)
    u = zeros(input_dimension(car))
    u[[end-1, end]] .= 1
    return u
end

function steering_mask(car::Mechanism)
    u = zeros(input_dimension(car))
    u[7] = 1
    return u
end

car = get_car(μ = 0.6)
initialize_car!(car, 0.0, -7.0)

K = 1000
storage = Storage(K, length(car.bodies))

drive_power = -0.3

τ = torque_mask(car)
θ = steering_mask(car)

for k in 1:K

    u = zeros(input_dimension(car))

    if iseven(k ÷ 20)
        u .+= -0.3 * θ
    else
        u .+= 0.3 * θ
    end

    if k > 30
        u .+= τ * drive_power
    end

    @time step!(car, get_maximal_state(car), u)
    Dojo.save_to_storage!(car, storage, k)
end

vis = Visualizer()
open(vis)
visualize(car, storage, vis = vis)