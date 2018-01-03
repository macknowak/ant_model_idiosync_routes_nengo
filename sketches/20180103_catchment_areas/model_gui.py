"""View representation in catchment areas.

This script provides a Nengo model that uses semantic pointers to represent
views perceived by the robot within catchment areas. Views are mimicked in such
a way that there is a unique view related to each waypoint and it is
represented as a semantic pointer of arbitrary numbers. Upon entering the
catchment area of a waypoint, the semantic pointer representing the
corresponding view is retrieved and scaled depending on the remaining distance
to the waypoint. For the robot to move, its wheel speeds must be set manually
using Nengo GUI.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import nengo.spa as spa
import numpy as np

with_vrep = False
if with_vrep:
    import vrepsim as vrs
else:
    waypts_pos = [[-0.5, 0.0], [1.5, 0.0]]
    waypts_names = ["Waypoint1", "Waypoint2"]
    bot_init_pos = [-2.0, 0.0]
    bot_init_orient_deg = 0

# --- PARAMETERS ---
class Params(object): pass
params = Params()

if with_vrep:
    # Waypoint parameters
    params.waypts_coll_name = "Waypoints"

    # Robot parameters
    params.bot_name = "Pioneer_p3dx"
    params.bot_motors_names = ["Pioneer_p3dx_leftMotor",
                               "Pioneer_p3dx_rightMotor"]

    # V-REP remote API parameters
    params.vrep_ip = '127.0.0.1'
    params.vrep_port = 19997

    # Simulation parameters
    params.nengo_sim_dt = 0.001  # s

# Waypoint parameters
params.catch_area_radius = 0.5

# Nengo model parameters
params.sp_dim = 16
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

    # Retrieve positions of waypoints
    if with_vrep:
        waypts_coll = vrs.Collection(vrep_sim, params.waypts_coll_name)
        waypts_pos = np.array(waypts_coll.get_positions(), dtype=float)[:,:2]
    else:
        waypts_pos = np.array(waypts_pos, dtype=float)

    # Validate number of waypoints
    n_waypts = len(waypts_pos)
    assert n_waypts >= 2, "Number of waypoints must not be less than 2."

    # Retrieve names of waypoints
    if with_vrep:
        waypts_names = waypts_coll.get_names()

    # Validate radius of catchment areas
    min_waypt_dist = 2 * params.catch_area_radius
    for w, waypt1_pos in enumerate(waypts_pos):
        for waypt2_pos in waypts_pos[w+1:]:
            waypts_dist = np.sqrt(((waypt1_pos - waypt2_pos)**2).sum())
            assert waypts_dist >= min_waypt_dist, \
                ("Radius of catchment areas must be less than {}."
                 "".format(waypts_dist / 2))

    # Create representation of the robot
    if with_vrep:
        bot = vrs.PioneerBot(vrep_sim, params.bot_name, None,
                             params.bot_motors_names)

    # Create vocabulary
    views_names = ["VIEW_"+waypt_name.upper() for waypt_name in waypts_names]
    views_vocab = spa.Vocabulary(params.sp_dim)
    views_vocab.extend(views_names)

    # Create communicator for data exchange with V-REP
    if with_vrep:
        sim_dt_ratio = int(sim_dt_ratio)
        vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    model = spa.SPA()
    with model:
        # Create node representing communicator for data exchange with V-REP
        if with_vrep:
            vrep_proxy = nengo.Node(vrep_comm, size_in=2, size_out=3)

        # Create node representing robot position
        if with_vrep:
            bot_pos_inp = nengo.Node(None, size_in=2)
            vrep_comm.add_output(lambda: bot.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[0:2], bot_pos_inp, synapse=None)
        else:
            bot_pos_inp = nengo.Node(bot_init_pos)

        # Create node representing robot orientation
        if with_vrep:
            bot_orient_inp = nengo.Node(None, size_in=1)
            vrep_comm.add_output(lambda: bot.get_orientation()[2], 1)
            nengo.Connection(vrep_proxy[2], bot_orient_inp, synapse=None)
        else:
            bot_orient_inp = nengo.Node(bot_init_orient_deg * np.pi / 180.0)

        # Create node representing view in catchment area
        def update_catch_area_view(t, x):
            waypts_dists = np.sqrt(((x - waypts_pos)**2).sum(axis=1))
            min_waypt_dist = waypts_dists.min()
            if min_waypt_dist < params.catch_area_radius:
                view_name = views_names[waypts_dists.argmin()]
                gain = 1.0 - (min_waypt_dist / params.catch_area_radius)
                return gain * views_vocab[view_name].v
            else:
                return [0.0] * params.sp_dim

        catch_area_view_inp = nengo.Node(update_catch_area_view, size_in=2,
                                         size_out=params.sp_dim)
        nengo.Connection(bot_pos_inp, catch_area_view_inp, synapse=None)

        # Create state representing current view
        model.view = spa.State(params.sp_dim, vocab=views_vocab)
        nengo.Connection(catch_area_view_inp, model.view.input)

        # Create node representing wheel speeds
        wheels_speeds_outp = nengo.Node([0], size_out=1)
        if with_vrep:
            vrep_comm.add_input(bot.wheels.set_velocities, 2)
            nengo.Connection(wheels_speeds_outp, vrep_proxy[0:2],
                             transform=[[1.0], [1.0]])

    # Start V-REP simulation in synchronous operation mode
    if with_vrep:
        vrep_sim.start_sim(options.verbose)

except Exception:
    if with_vrep:
        vrep_sim.disconnect(options.verbose)
    raise
