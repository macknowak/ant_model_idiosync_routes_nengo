"""Robot displacement during simulation.

This script provides a Nengo model that controls robot forward motion as well
as a mechanism that suddenly displaces the robot to an arbitrary location
during the simulation. The robot can be displaced either when stopped or when
in motion.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import vrepsim as vrs

# --- PARAMETERS ---
class Params(object): pass
params = Params()

# Location parameters
params.bot_displacement_pos = [-2.0, 1.0]

# Robot parameters
params.bot_name = "Pioneer_p3dx"
params.bot_motors_names = ["Pioneer_p3dx_leftMotor", "Pioneer_p3dx_rightMotor"]
params.bot_dynamic_objs_names = [
    "Pioneer_p3dx_leftMotor", "Pioneer_p3dx_leftWheel",
    "Pioneer_p3dx_rightMotor", "Pioneer_p3dx_rightWheel",
    "Pioneer_p3dx_caster_freeJoint1","Pioneer_p3dx_caster_link",
    "Pioneer_p3dx_caster_freeJoint2", "Pioneer_p3dx_caster_wheel"
    ]

# V-REP remote API parameters
params.vrep_ip = '127.0.0.1'
params.vrep_port = 19997

# Simulation parameters
params.nengo_sim_dt = 0.001  # s
params.sim_cycle_duration = 0.5  # s
params.sim_duration = 5.0  # s

# Other parameters
params.stop_bot_before_move = True
# ------------------

# --- OPTIONS ---
class Namespace(object): pass
options = Namespace()
options.verbose = True
# ---------------

# Connect to V-REP
vrep_sim = vrs.Simulator(params.vrep_ip, params.vrep_port)
vrep_sim.connect(verbose=options.verbose)

try:
    # Validate V-REP simulation time step
    vrep_sim_dt = vrep_sim.get_sim_dt()
    sim_dt_ratio = vrep_sim_dt / params.nengo_sim_dt
    assert sim_dt_ratio == round(sim_dt_ratio), \
        ("V-REP simulation time step is not evenly divisible by Nengo "
         "simulation time step.")

    # Create representation of the robot
    bot = vrs.PioneerBot(vrep_sim, params.bot_name, None,
                         params.bot_motors_names)

    # Create representations of dynamically simulated child objects of the
    # robot
    bot_dynamic_objs = [vrs.SceneObject(vrep_sim, name)
                        for name in params.bot_dynamic_objs_names]

    # Create communicator for data exchange with V-REP
    sim_dt_ratio = int(sim_dt_ratio)
    vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    model = nengo.Network()
    with model:
        # Create node representing communicator for data exchange with V-REP
        vrep_proxy = nengo.Node(vrep_comm, size_in=2, size_out=0)

        # Create node representing wheel speeds
        def update_wheel_speeds(t):
            if params.stop_bot_before_move:
                if (t < params.sim_duration - 1.0
                    or t > params.sim_duration + 1.0):
                    return [5.0, 5.0]
                else:
                    return [0.0, 0.0]
            else:
                return [5.0, 5.0]

        wheel_speeds = nengo.Node(update_wheel_speeds, size_in=0)
        vrep_comm.add_input(bot.wheels.set_velocities, 2)
        nengo.Connection(wheel_speeds[0:2], vrep_proxy[0:2], synapse=None)

    # Start V-REP simulation in synchronous operation mode
    vrep_sim.start_sim(options.verbose)

    try:
        # Run simulation
        nengo_sim = nengo.Simulator(model, dt=params.nengo_sim_dt)
        with nengo_sim:
            # Run the first cycle of simulation (before displacement)
            while nengo_sim.time < params.sim_duration:
                nengo_sim.run(params.sim_cycle_duration)

            # Set positions of dynamically simulated child objects of the robot
            # to those that they already hold; this is a workaround to force
            # dynamic reset of those objects (that is to remove them from the
            # dynamics engine) as otherwise they could not be moved along with
            # the robot (they will be added again to the dynamics engine once
            # the robot has been moved to the new position)
            for dynamic_obj in bot_dynamic_objs:
                dynamic_obj.set_position(dynamic_obj.get_position(),
                                         allow_in_sim=True)

            # Move robot to the new position
            new_bot_pos = params.bot_displacement_pos + [bot.get_position()[2]]
            bot.set_position(new_bot_pos, allow_in_sim=True)

            # Run the second cycle of simulation (after displacement)
            while nengo_sim.time < 2 * params.sim_duration:
                nengo_sim.run(params.sim_cycle_duration)

    finally:
        # Stop V-REP simulation
        vrep_sim.stop_sim(options.verbose)

finally:
    # Disconnect from V-REP
    vrep_sim.disconnect(options.verbose)
