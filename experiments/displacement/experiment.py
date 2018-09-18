"""Navigation to a goal interrupted by displacements.

This script provides implementation of an experiment in which the Nengo model
of ant navigation along established idiosyncratic routes controls a
differential-drive robot simulated using V-REP simulator such that it navigates
towards a specified goal in the scene, but upon arriving at a particular
location, the robot is displaced to a different release point. Communication
between the Nengo model and V-REP is handled using the V-REP remote API.

The robot operates in a scene that must contain several goals linked by routes.
The routes can possibly be multi-segment, with successive segments connected at
waypoints. The robot model used in V-REP is Pioneer_p3dx. It has several
sensors, 2 motor-driven wheels (1 on the left and 1 on the right), and 1 caster
wheel (in the back). None of the sensors are used, however, whereas each
motor-driven wheel receives signals controlling its velocity from the Nengo
model. The initial position and orientation of the robot can optionally be set
prior to simulation. In the beginning of simulation, a current goal is set and
then the robot sets off trying to reach that goal. However, there is one
specific location in the scene such that if the robot happens to arrive within
a specified radius of that location, it is immediately displaced to a release
point. During a simulation, multiple displacements may occur from that location
as release points are specified in a sequence and employed one by one on
subsequent arrivals (there may be many different release points or many
instances of the same release point; the maximum number of displacements
depends on the length of that sequence). Release points, apart from location,
may optionally define the subsequent robot orientation.

To successfully run this script, V-REP simulator must first be launched with a
continuous remote API server service started and then scene file 'scene.ttt'
must be opened in the simulator. To allow this script to remotely control the
Pioneer_p3dx robot that is part of this scene, the original child script
associated with this robot was removed from the scene file.

Moreover, to successfully run this script, the following V-REP files (or,
alternatively, links to them) have to exist in the current directory so that
the V-REP remote API could be used (here, 'VREP_DIR' denotes the directory in
which V-REP is installed):

- 'vrep.py' (original file in: 'VREP_DIR/programming/remoteApiBindings/python/
  python/');
- 'vrepConst.py' (original file in: 'VREP_DIR/programming/remoteApiBindings/
  python/python/');
- ['remoteApi.dll' | 'remoteApi.dylib' | 'remoteApi.so'] (original file in:
  'VREP_DIR/programming/remoteApiBindings/lib/lib/[32Bit | 64Bit]/').

This script was developed for Nengo 2.7.0 and V-REP 3.4.0.
"""

__version__ = '1.0'
__author__ = "Przemyslaw (Mack) Nowak"

import argparse
import datetime
import os
import platform
import warnings

import numpy as np
import simtools
import vrepsim as vrs

from model import Model, __version__ as model_version
from vrepscene import Scene

# Filenames
params_filename = "param.json"
platform_filename = "platform.json"
versions_filename = "version.json"

# Parameters to be saved
saved_params = [
    'experiment_version', 'model_version', 'sim_id',
    'bot_displacept_dist_thres', 'bot_displacept_pos', 'action_n2pr_coeff_gvr',
    'action_v2pr_coeff_gvr', 'action_v2pr_coeff_intercept', 'bot_init_orient',
    'bot_init_pos', 'bot_releasepts_orients', 'bot_releasepts_pos',
    'catch_area_radius', 'catch_vec_core', 'catch_vec_thres', 'goal',
    'init_global_vec', 'nengo_backend', 'nengo_sim_dt', 'np_seed', 'prb_syn',
    'scene_filename', 'scene_radius', 'sim_cycle_duration', 'sim_duration',
    'sp_dim', 'sp_max_similarity', 'vrep_dyn_eng_dt', 'vrep_dyn_eng_name',
    'vrep_sim_dt'
    ]

# Process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v", "--verbose",
    dest='verbose', action='store_true',
    help="display information on performed operations")
options = simtools.parse_args(
    ['params_filename', 'sim_id', 'save_data', 'data_dirname'], parser=parser)

# Load parameters
assert options.params_filename is not None, "Parameter file is not specified."
params = simtools.load_params(options.params_filename)

# Validate experiment version
assert __version__ == params.experiment_version, \
    ("Experiment version in the parameter file does not match the actual "
     "experiment version.")

# Validate Nengo backend
assert params.nengo_backend in ('nengo', 'nengo_ocl'), \
    "Backend '{}' is not supported.".format(params.nengo_backend)

# Import appropriate Nengo backend
if params.nengo_backend == 'nengo':
    import nengo
elif params.nengo_backend == 'nengo_ocl':
    import nengo_ocl
    import pyopencl as cl

# Validate option to save simulation data
if options.save_data:
    assert params.saved_data, "Simulation data to be saved is not specified."
else:
    assert not params.saved_data, \
        "Simulation data cannot be saved due to missing option to save data."

# Add model version to parameters
params.model_version = model_version

# Add simulation id to parameters
params.sim_id = options.sim_id

# Set random seed
if params.np_seed is None:
    params.np_seed = simtools.generate_seed(4)
np.random.seed(params.np_seed)

# Validate simulation duration
n_sim_cycles = params.sim_duration / params.sim_cycle_duration
assert n_sim_cycles == round(n_sim_cycles), \
    "Simulation duration is not evenly divisible by simulation cycle duration."

# Retrieve number of displacements
n_displaces = len(params.bot_releasepts_pos)

# Validate displacements
assert n_displaces >= 1, "Number of displacements is less than 1."
assert len(params.bot_releasepts_orients) == n_displaces, \
    ("Number of new positions due to displacements does not match the number "
     "of new orientations.")

# Connect to V-REP
vrep_sim = vrs.Simulator(params.vrep_ip, params.vrep_port)
vrep_sim.connect(verbose=options.verbose)

model = None
try:
    # Validate V-REP simulation time step
    params.vrep_sim_dt = vrep_sim.get_sim_dt()
    sim_dt_ratio = params.vrep_sim_dt / params.nengo_sim_dt
    assert sim_dt_ratio == round(sim_dt_ratio), \
        ("V-REP simulation time step is not evenly divisible by Nengo "
         "simulation time step.")

    # Validate scene filename
    vrep_scene_filename = os.path.basename(vrep_sim.get_scene_path())
    assert params.scene_filename == vrep_scene_filename, \
        "Scene filename does not match the actual scene filename."

    # Create representation of the scene
    scene = Scene(vrep_sim)

    # Validate current goal
    assert params.goal in scene.goals_names, \
        "Goal '{}' is invalid.".format(params.goal)

    # Validate initial global vector
    if params.init_global_vec is not None:
        if params.bot_init_pos is not None:
            if params.init_global_vec != params.bot_init_pos:
                warnings.warn("Initial global vector and initial robot "
                              "position are different.")
        else:
            bot_init_pos = [round(p, 2) for p in scene.bot.get_position()[:2]]
            if params.init_global_vec != bot_init_pos:
                warnings.warn("Initial global vector and initial robot "
                              "position are different.")

    # Update robot initial position
    if params.bot_init_pos is not None:
        scene.bot.set_position(
            params.bot_init_pos+[scene.bot.get_position()[2]])
    else:
        params.bot_init_pos = scene.bot.get_position()[:2]

    # Update robot initial orientation
    if params.bot_init_orient is not None:
        scene.bot.set_orientation(
            scene.bot.get_orientation()[:2]+[params.bot_init_orient])
    else:
        params.bot_init_orient = scene.bot.get_orientation()[2]

    # If necessary, update initial global vector
    if params.init_global_vec is None:
        params.init_global_vec = scene.bot.get_position()[:2]

    # Add V-REP dynamics engine name to parameters
    params.vrep_dyn_eng_name = vrep_sim.get_dyn_eng_name()

    # Add V-REP dynamics engine time step to parameters
    params.vrep_dyn_eng_dt = vrep_sim.get_dyn_eng_dt()

    # Create communicator for data exchange with V-REP
    sim_dt_ratio = int(sim_dt_ratio)
    vrep_comm = vrs.nengo.NengoComm(vrep_sim, sim_dt_ratio)

    # Create Nengo model controlling the robot
    model = Model(params, scene, vrep_comm)

    # Set current goal
    model.set_goal(params.goal)

    # Start V-REP simulation in synchronous operation mode
    vrep_sim.start_sim(options.verbose)

    try:
        # Run simulation for the specified time or until the goal is reached
        if options.verbose:
            start_time = datetime.datetime.now().replace(microsecond=0)
        if params.nengo_backend == 'nengo':
            nengo_sim = nengo.Simulator(model, dt=params.nengo_sim_dt)
        elif params.nengo_backend == 'nengo_ocl':
            nengo_sim = nengo_ocl.Simulator(model, dt=params.nengo_sim_dt)
        if options.verbose:
            end_time = datetime.datetime.now().replace(microsecond=0)
            sim_build_time = end_time - start_time
            start_time = end_time
        bot_displacept_pos = np.array(params.bot_displacept_pos, dtype=float)
        displace_no = 0
        with nengo_sim:
            while nengo_sim.time < params.sim_duration:
                # If the robot reports reaching the goal, stop the wheels and
                # stop moving
                if model.goal_reached:
                    scene.bot.wheels.set_velocities((0.0, 0.0))
                    if options.verbose:
                        end_time = \
                            datetime.datetime.now().replace(microsecond=0)
                        print("Goal '{}' was reached.".format(params.goal))
                    break

                # If the point of displacement has almost been reached and the
                # robot should be displaced, displace the robot to a release
                # position
                if displace_no < n_displaces:
                    bot_pos = scene.bot.get_position()[:2]
                    bot_displacept_dist = np.sqrt(
                        ((bot_displacept_pos - bot_pos)**2).sum())
                    if bot_displacept_dist < params.bot_displacept_dist_thres:
                        scene.move_bot(
                            params.bot_releasepts_pos[displace_no],
                            params.bot_releasepts_orients[displace_no])
                        model.on_displace()
                        displace_no += 1

                # Run a single cycle of simulation
                nengo_sim.run(params.sim_cycle_duration)
            else:
                # If the robot has not reported reaching the goal until the
                # time is up, stop the wheels and stop moving
                scene.bot.wheels.set_velocities((0.0, 0.0))
                if options.verbose:
                    end_time = datetime.datetime.now().replace(microsecond=0)
                    print("Time is up, goal '{}' was not reached."
                          "".format(params.goal))
            if options.verbose:
                sim_run_time = end_time - start_time
                bot_pos = scene.bot.get_position()[:2]
                goal_idx = scene.goals_names.index(params.goal)
                goal_pos = scene.goals_pos[goal_idx]
                bot_goal_dist = np.sqrt(((goal_pos - bot_pos)**2).sum())
                print("Simulation time: {:.3f} s.".format(nengo_sim.time))
                print("Distance to goal: {:.2f} m.".format(bot_goal_dist))
                print("Displacements: {}.".format(displace_no))
                print("Simulation build time: {}.".format(sim_build_time))
                print("Simulation run time: {}.".format(sim_run_time))

    finally:
        # Stop V-REP simulation
        vrep_sim.stop_sim(options.verbose)

    # If necessary, save data
    if options.save_data:
        if model is not None:
            # Save simulation data
            model.save_data(nengo_sim, options.data_dirname)

        # Save parameters
        params.save(params_filename, saved_params, sort_keys=True)

        # Save platform information
        simtools.save_platform(platform_filename)

        # Save software versions
        versions_info = [
            ('model', model_version),
            ('experiment', __version__),
            ('python', platform.python_version()),
            ('nengo', nengo.__version__),
            ('numpy', np.__version__),
            ('simtools', simtools.__version__),
            ('v-rep', vrep_sim.get_version()),
            ('vrepsim', vrs.__version__)
            ]
        if params.nengo_backend == 'nengo_ocl':
            versions_info.extend([
                ('nengo_ocl', nengo_ocl.__version__),
                ('pyopencl', cl.VERSION_TEXT)
                ])
            versions_info = versions_info[:3] + sorted(versions_info[3:])
        simtools.save_versions(versions_filename, versions_info)

finally:
    # Disconnect from V-REP
    vrep_sim.disconnect(options.verbose)
