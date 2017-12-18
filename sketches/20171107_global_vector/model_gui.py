"""Maintenance of global vector.

This script provides a Nengo model that maintains a global vector updated based
on the robot current and previous positions. The global vector is calculated
explicitly and accurately (without noise) and points from the initial robot
position to the current robot position.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo

with_vrep = False
if with_vrep:
    import vrepsim as vrs
else:
    bot_init_pos = [0.0, 0.0]

# --- PARAMETERS ---
class Params(object): pass
params = Params()

if with_vrep:
    # Robot parameters
    params.bot_name = "Pioneer_p3dx"
    params.bot_motors_names = ["Pioneer_p3dx_leftMotor",
                               "Pioneer_p3dx_rightMotor"]

    # V-REP remote API parameters
    params.vrep_ip = '127.0.0.1'
    params.vrep_port = 19997

    # Simulation parameters
    params.nengo_sim_dt = 0.001  # s
# ------------------

# --- OPTIONS ---
class Namespace(object): pass
options = Namespace()
options.verbose = True
# ---------------

# Connect to V-REP
if with_vrep:
    vrep_sim = vrs.Simulator(params.vrep_ip, params.vrep_port)
    vrep_sim.connect(verbose=options.verbose)

try:
    # Validate V-REP simulation time step
    if with_vrep:
        params.vrep_sim_dt = vrep_sim.get_sim_dt()
        sim_dt_ratio = params.vrep_sim_dt / params.nengo_sim_dt
        assert sim_dt_ratio == round(sim_dt_ratio), \
            ("V-REP simulation time step must be evenly divisible by Nengo "
             "simulation time step.")

    # Create representation of the robot
    if with_vrep:
        bot = vrs.PioneerBot(vrep_sim, params.bot_name, None,
                             params.bot_motors_names)

    # Create communicator for data exchange with V-REP
    if with_vrep:
        sim_dt_ratio = int(sim_dt_ratio)
        vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    model = nengo.Network()
    with model:
        # Create node representing communicator for data exchange with V-REP
        if with_vrep:
            vrep_proxy = nengo.Node(vrep_comm, size_in=2, size_out=2)

        # Create node representing robot position
        if with_vrep:
            bot_pos_inp = nengo.Node(None, size_in=2)
            vrep_comm.add_output(lambda: bot.get_position()[0:2], 2)
            nengo.Connection(vrep_proxy[0:2], bot_pos_inp, synapse=None)
        else:
            bot_pos_inp = nengo.Node(bot_init_pos)

        # Create node representing global vector
        global_vec = [0.0, 0.0]
        if with_vrep:
            bot_prev_pos = bot.get_position()[0:2]
        else:
            bot_prev_pos = bot_init_pos

        def update_global_vec(t, x):
            global global_vec
            global bot_prev_pos
            bot_diff_pos = [x[0] - bot_prev_pos[0], x[1] - bot_prev_pos[1]]
            if any(bot_diff_pos):
                global_vec[0] += bot_diff_pos[0]
                global_vec[1] += bot_diff_pos[1]
                bot_prev_pos[0] += bot_diff_pos[0]
                bot_prev_pos[1] += bot_diff_pos[1]
            return global_vec

        global_vec_inp = nengo.Node(update_global_vec, size_in=2, size_out=2)
        nengo.Connection(bot_pos_inp, global_vec_inp, synapse=None)

        # Create node representing wheel speeds
        def update_wheel_speeds(t):
            if with_vrep:
                if t < 10.0:
                    return [1.5, 2.5]
                else:
                    return [0.0, 0.0]
            else:
                return [0.0, 0.0]

        wheel_speeds = nengo.Node(update_wheel_speeds, size_in=0, size_out=2)
        if with_vrep:
            vrep_comm.add_input(bot.wheels.set_velocities, 2)
            nengo.Connection(wheel_speeds, vrep_proxy[0:2])

    # Start V-REP simulation in synchronous operation mode
    if with_vrep:
        vrep_sim.start_sim(options.verbose)

except Exception:
    if with_vrep:
        vrep_sim.disconnect(options.verbose)
    raise
