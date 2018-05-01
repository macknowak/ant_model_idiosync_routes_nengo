"""Robot motion controller.

This script provides a Nengo model of a simple proportional controller for
controlling motion of the model of the Pioneer P3-DX robot provided by V-REP.

The controller implements a "go-to-goal" behavior. It also assumes that the
positions of the goal and of the robot are accurately known at each time
instant.

The controller operates according to the following rules:

1. If the distance from the robot to the goal is less than or equal to a
   predefined margin, the goal is assumed to have been reached and the robot
   stops.
2. Otherwise, two cases are distinguished:
    - if the angle between the robot and the goal is greater than a predefined
      threshold, the robot rotates on the spot towards the goal;
    - otherwise the robot moves towards the goal in a curvilinear manner.

During a rotation on the spot, wheel velocities are computed by scaling the
default wheel speed proportionally to the angle between the robot and the goal;
in this case both wheels move at the same speed but in opposite directions,
with the actual directions determined based on the direction to the goal.

During curvilinear motion forward, each wheel velocity is computed by
adjusting the default wheel speed such that the angle between the robot and
the goal is either added to or subtracted from the default wheel speed,
which depends on the direction to the goal.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import numpy as np
import simtools

with_vrep = False
if with_vrep:
    import vrepsim as vrs
else:
    bot_init_pos = [-0.5, 1.0]
    bot_init_orient_deg = -90
    goal_init_pos = [0.0, 0.0]

# --- PARAMETERS ---
class Params(object): pass
params = Params()

if with_vrep:
    # Goal parameters
    params.goal_name = "Goal"

    # Robot parameters
    params.bot_name = "Pioneer_p3dx"
    params.bot_motors_names = ["Pioneer_p3dx_leftMotor",
                               "Pioneer_p3dx_rightMotor"]

    # V-REP remote API parameters
    params.vrep_ip = '127.0.0.1'
    params.vrep_port = 19997

# Motion parameters
params.default_wheel_speed = 5.0  # rad/s
params.theta_thres = 0.4  # rad
params.goal_dist_margin = 0.05  # m

# Simulation parameters
params.nengo_sim_dt = 0.001  # s

# Random number generator parameters
params.np_seed = None
# ------------------

# --- OPTIONS ---
class Namespace(object): pass
options = Namespace()
options.verbose = True
# ---------------

# Set random seed
if params.np_seed is None:
    params.np_seed = simtools.generate_seed(4)
np.random.seed(params.np_seed)
print("NumPy seed: {}".format(params.np_seed))

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

    # Create representation of the goal
    if with_vrep:
        goal = vrs.Dummy(vrep_sim, params.goal_name)

    # Create communicator for data exchange with V-REP
    if with_vrep:
        sim_dt_ratio = int(sim_dt_ratio)
        vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    model = nengo.Network()
    with model:
        # Create node representing communicator for data exchange with V-REP
        if with_vrep:
            vrep_proxy = nengo.Node(vrep_comm, size_in=2, size_out=5)

        # Create node representing robot position
        if with_vrep:
            bot_pos_inp = nengo.Node(None, size_in=2)
            vrep_comm.add_output(lambda: bot.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[0:2], bot_pos_inp, synapse=None)
        else:
            bot_pos_inp = nengo.Node(bot_init_pos)

        # Create node representing robot orientation
        if with_vrep:
            bot_orient_inp = nengo.Node(lambda t, x: x, size_in=1, size_out=1)
            vrep_comm.add_output(lambda: bot.get_orientation()[2], 1)
            nengo.Connection(vrep_proxy[2], bot_orient_inp, synapse=None)
        else:
            bot_orient_inp = nengo.Node(bot_init_orient_deg * np.pi / 180.0)

        # Create node representing goal position
        if with_vrep:
            goal_pos_inp = nengo.Node(None, size_in=2)
            vrep_comm.add_output(lambda: goal.get_position()[:2], 2)
            nengo.Connection(vrep_proxy[3:5], goal_pos_inp, synapse=None)
        else:
            goal_pos_inp = nengo.Node(goal_init_pos)

        # Create node representing motion vector
        def update_motion_vec(t, x):
            goal_vec = x[3:5] - x[0:2]
            c, s = np.cos(-x[2]), np.sin(-x[2])
            goal_rot = np.array([[c, -s], [s, c]], dtype=float)
            return np.dot(goal_rot, goal_vec)

        motion_vec_inp = nengo.Node(update_motion_vec, size_in=5, size_out=2)
        nengo.Connection(bot_pos_inp, motion_vec_inp[0:2], synapse=None)
        nengo.Connection(bot_orient_inp, motion_vec_inp[2], synapse=None)
        nengo.Connection(goal_pos_inp, motion_vec_inp[3:5], synapse=None)

        # Create node representing wheel speeds
        def update_wheel_speeds(t, x):
            if np.sqrt((x**2).sum()) < params.goal_dist_margin:
                return [0.0, 0.0, 0.0]  # stop
            theta = np.arctan2(x[1], x[0])
            if -params.theta_thres < theta < params.theta_thres:
                return [params.default_wheel_speed - theta,
                        params.default_wheel_speed + theta,
                        1.0]  # (curvilinear) motion forward
            else:
                return [-params.default_wheel_speed * theta,
                        params.default_wheel_speed * theta,
                        -1.0]  # rotation on the spot

        wheel_speeds = nengo.Node(update_wheel_speeds, size_in=2, size_out=3)
        nengo.Connection(motion_vec_inp, wheel_speeds,
                         synapse=params.nengo_sim_dt)
        if with_vrep:
            vrep_comm.add_input(bot.wheels.set_velocities, 2)
            nengo.Connection(wheel_speeds[0:2], vrep_proxy[0:2], synapse=None)

        # Start V-REP simulation in synchronous operation mode
        if with_vrep:
            vrep_sim.start_sim(options.verbose)

except Exception:
    if with_vrep:
        vrep_sim.disconnect(options.verbose)
    raise
