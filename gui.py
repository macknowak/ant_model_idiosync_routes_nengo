"""Ant navigation along established idiosyncratic routes, controlled by Nengo.

This script provides support for simulating the Nengo model of ant navigation
along established idiosyncratic routes using Nengo GUI. The Nengo model can
either be coupled to a differential-drive robot simulated using V-REP simulator
or run independently. Communication between the Nengo model and V-REP is
handled using the V-REP remote API.

This script was developed for Nengo 2.7.0, Nengo GUI 0.3.1, and V-REP 3.4.0.
"""

__version__ = '1.0'
__author__ = "Przemyslaw (Mack) Nowak"

import os

import numpy as np; reload(np)

import model; reload(model)
from model import Model

with_vrep = False
if with_vrep:
    from vrepscene import Scene
    import vrepsim as vrs
else:
    from dummyscene import Scene

# --- PARAMETERS ---
class Params(object): pass
params = Params()

if with_vrep:
    # V-REP remote API parameters
    params.vrep_ip = '127.0.0.1'
    params.vrep_port = 19997

    # Simulation parameters
    params.nengo_sim_dt = 0.001  # s

else:
    # Location parameters
    params.bot_init_pos = [0.0, 0.0]
    params.bot_init_orient = 0.0

# Location parameters
params.scene_radius = 10.0
params.catch_area_radius = 1.0
params.catch_vec_thres = 0.2
params.catch_vec_core = 0.2
params.goal = "Feed1"
params.init_global_vec = [0.0, 0.0]

# Nengo model parameters
params.sp_dim = 64
params.sp_max_similarity = 0.1
params.action_n2pr_coeff_gvr = 0.2
params.action_v2pr_coeff_gvr = 0.5
params.action_v2pr_coeff_intercept = 0.2

# Random number generator parameters
params.np_seed = 2062361305

# Other parameters
params.saved_data = None
# ------------------

# --- OPTIONS ---
class Namespace(object): pass
options = Namespace()
options.verbose = True
# ---------------

# Set random seed
if params.np_seed is None:
    params.np_seed = int(os.urandom(4).encode('hex'), 16)
np.random.seed(params.np_seed)

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

    # Create representation of the scene
    if with_vrep:
        scene = Scene(vrep_sim)
    else:
        scene = Scene(params)

    # Create communicator for data exchange with V-REP
    if with_vrep:
        sim_dt_ratio = int(sim_dt_ratio)
        vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    if with_vrep:
        model = Model(params, scene, vrep_comm)
    else:
        model = Model(params, scene)

    # Set current goal
    model.set_goal(params.goal)

    # Start V-REP simulation in synchronous operation mode
    if with_vrep:
        vrep_sim.start_sim(options.verbose)

except Exception:
    if with_vrep:
        vrep_sim.disconnect(options.verbose)
    raise
