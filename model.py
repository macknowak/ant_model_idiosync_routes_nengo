"""Ant navigation along established idiosyncratic routes, controlled by Nengo.

This module provides a Nengo model that controls a differential-drive robot
simulated using V-REP simulator, which mimics a desert ant navigating in a
foraging area with several established routes between different goals.
Communication between the Nengo model and V-REP is handled using the V-REP
remote API.

The robot is supposed to navigate to a particular goal among several ones that
exist in the scene. The scene must also contain routes linking those goals. The
routes can possibly be multi-segment, with successive segments connected at
waypoints. Navigation is based on three kinds of vectors: (1) the global vector
(which points from the current robot position to the origin), (2) local vectors
(each related to a waypoint on a route and pointing to the next waypoint on
that route), and (3) catchment vectors (which point from the current robot
position to the waypoint they are related to). The robot can either rely on
global vector navigation (e.g., when it is in an unfamiliar area) or follow a
route using local vectors. Catchment vectors are employed only when the robot
enters the catchment area of a waypoint and are activated during both global
vector navigation and route following. They imitate guidance based on comparing
the current view with the stored view related to that waypoint as vision is
implemented only in an abstract way. The contributions of the three vectors are
dynamically weighted. In the beginning of simulation, a current goal should be
set (it can be changed at any time). Initially, the robot relies on global
vector navigation, but can switch to route following whenever it encounters a
waypoint belonging to a route that leads to the current goal (which may be
already at the start). As the robot moves, arrivals at waypoints are detected
as is arrival at the goal. Additionally, reaction to a passive displacement can
occur, which involves switching to global vector navigation.

The Nengo model comprises the following core objects:

- a node representing robot position (obtained from V-REP);
- a node representing robot orientation (obtained from V-REP);
- a node representing global vector (calculated directly);
- a node representing catchment area (that identifies the relevant waypoint and
  reports the normalized distance to that waypoint; calculated directly);
- a node representing catchment vector (along with its gain; calculated
  directly);
- a node representing vision (calculated directly);
- a node representing displacement (calculated directly);
- a node representing new goal (calculated directly);
- a state representing current goal (as a semantic pointer);
- a node representing goal detection (calculated directly);
- an associative memory between goals and goal vectors;
- an ensemble representing target vector;
- a gate for inhibiting target vector;
- a gate for inhibiting target vector depending on whether current goal is set;
- an ensemble representing normalized target vector;
- a state representing current view (as a semantic pointer);
- a binder for current goal and current view;
- a state representing currently viewed routepoint (as a semantic pointer);
- a gate for inhibiting currently viewed routepoint;
- a state representing previous routepoint (as a semantic pointer);
- a cleanup memory for routepoints;
- an associative memory between previous and next routepoints;
- a state representing next routepoint (as a semantic pointer);
- a gate for inhibiting cleanup memory for routepoints;
- a network transiently inhibiting cleanup memory for routepoints;
- an associative memory between routepoints and views;
- a comparator between current and retrieved views;
- an associative memory between routepoints and normalized local vectors
  between route waypoints;
- an ensemble representing gain of the catchment vector;
- a network representing scaled catchment vector;
- a network representing scaled local vector;
- a network representing scaled target vector;
- a network aggregating scaled vectors and transforming them to the robot
  reference frame;
- an ensemble representing motion vector;
- a node representing wheel speeds (sent to V-REP; calculated directly);
- a gate for reset signal;
- an action selection circuit.

Although the robot model used in V-REP may contain sensors, they are not used
because the robot position is obtained directly and vision is implemented in an
abstract way, being imitated based on that position by representing views as
high-dimensional vectors of arbitrary numbers. The robot model must comprise
2 motor-driven wheels (1 on the left and 1 on the right), each of which
receives a signal controlling its velocity from the Nengo model. The
recommended robot model offered by V-REP is Pioneer_p3dx.

This module was developed for Nengo 2.7.0 and V-REP 3.4.0.
"""

__version__ = '1.0'
__author__ = "Przemyslaw (Mack) Nowak"

import os
import warnings

import nengo
import nengo.networks as networks
import nengo.presets as presets
import nengo.spa as spa
from nengo.utils.network import with_self
import numpy as np


class Model(spa.SPA):
    """Nengo model of ant navigation along established idiosyncratic routes."""

    probeable = (
        'action_names', 'bg', 'bot_orient', 'bot_pos', 'catch_area',
        'catch_vec', 'displace', 'gain_catch_vec', 'gain_catch_vec_neurons',
        'gate_reset', 'gate_routept_cleanup', 'gate_target_vec',
        'gate_target_vec_goal', 'gate_view_routept', 'global_vec', 'goal',
        'goal_neurons', 'goal0view', 'goal_detect', 'goal_input',
        'goals2goals_vecs', 'goals_names', 'goals_vocab', 'motion_vec',
        'motion_vec_neurons', 'next_routept', 'next_routept_neurons',
        'norm_target_vec', 'prev2next_routepts', 'prev_routept',
        'prev_routept_neurons', 'routept_cleanup', 'routepts2norm_local_vecs',
        'routepts2views', 'routepts_vocab', 'routes_waypts_names',
        'routes_waypts_pos', 'scaled_catch_vec', 'scaled_local_vec',
        'scaled_target_vec', 'target_vec', 'target_vec_neurons', 'thal',
        'thal_neurons', 'view', 'view_neurons', 'view7view', 'view_routept',
        'view_routept_neurons', 'views_vocab', 'vision', 'waypts_names',
        'waypts_pos', 'wheel_speeds'
        )
    action_names_filename = "actionname.txt"
    bg_filename = "bg.txt"
    bot_orient_filename = "botorient.txt"
    bot_pos_filename = "botpos.txt"
    catch_area_filename = "catcharea.txt"
    catch_vec_filename = "catchvec.txt"
    displace_filename = "displace.txt"
    gain_catch_vec_filename = "gaincatchvec.txt"
    gain_catch_vec_neurons_filename = "gaincatchvecnrn.npz"
    gate_reset_filename = "gatereset.txt"
    gate_routept_cleanup_filename = "gaterouteptcleanup.txt"
    gate_target_vec_filename = "gatetargetvec.txt"
    gate_target_vec_goal_filename = "gatetargetvecgoal.txt"
    gate_view_routept_filename = "gateviewroutept.txt"
    global_vec_filename = "globalvec.txt"
    goal_filename = "goal.txt"
    goal_neurons_filename = "goalnrn.npz"
    goal0view_filename = "goal0view.txt"
    goal_detect_filename = "goaldetect.txt"
    goal_input_filename = "goalinput.txt"
    goals2goals_vecs_filename = "goal2goalvec.txt"
    goals_names_filename = "goalname.txt"
    goals_vocab_filename = "goalvocab.txt"
    motion_vec_filename = "motionvec.txt"
    motion_vec_neurons_filename = "motionvecnrn.npz"
    next_routept_filename = "nextroutept.txt"
    next_routept_neurons_filename = "nextrouteptnrn.npz"
    norm_target_vec_filename = "normtargetvec.txt"
    prev2next_routepts_filename = "prev2nextroutept.txt"
    prev_routept_filename = "prevroutept.txt"
    prev_routept_neurons_filename = "prevrouteptnrn.npz"
    routept_cleanup_filename = "routeptcleanup.txt"
    routepts2norm_local_vecs_filename = "routept2normlocalvec.txt"
    routepts2views_filename = "routept2view.txt"
    routepts_vocab_filename = "routeptvocab.txt"
    routes_waypts_names_filename = "routewayptname.txt"
    routes_waypts_pos_filename = "routewayptpos.txt"
    scaled_catch_vec_filename = "scaledcatchvec.txt"
    scaled_local_vec_filename = "scaledlocalvec.txt"
    scaled_target_vec_filename = "scaledtargetvec.txt"
    target_vec_filename = "targetvec.txt"
    target_vec_neurons_filename = "targetvecnrn.npz"
    thal_filename = "thal.txt"
    thal_neurons_filename = "thalnrn.npz"
    view_filename = "view.txt"
    view_neurons_filename = "viewnrn.npz"
    view7view_filename = "view7view.txt"
    view_routept_filename = "viewroutept.txt"
    view_routept_neurons_filename = "viewrouteptnrn.npz"
    views_vocab_filename = "viewvocab.txt"
    vision_filename = "vision.txt"
    wheel_speeds_filename = "wheelspeed.txt"
    waypts_names_filename = "wayptname.txt"
    waypts_pos_filename = "wayptpos.txt"

    def __init__(self, params, scene, vrep_comm=None):
        super(Model, self).__init__()
        self.params = params
        self.scene = scene
        self.vrep_comm = vrep_comm
        self._goal_reached = False
        self._goal_name = None
        self._new_goal_name = "NONE"
        self._displaced = False
        self.validate_params(self.params, self.scene)
        self._make_model()
        if self.params.saved_data:
            self._make_probes()

    @property
    def goal_reached(self):
        """Flag indicating whether goal is reached."""
        return self._goal_reached

    @classmethod
    def validate_params(cls, params, scene):
        """Validate parameters."""

        # Validate number of goals
        assert len(scene.goals_names) >= 2, "Number of goals is less than 2."

        # Validate names of goals
        for goal_name in scene.goals_names:
            assert goal_name.upper() != "NONE", \
                "Goal name '{}' is not allowed.".format(goal_name)

        # Validate radius of catchment areas
        min_waypts_dist = np.sqrt(
            min(((pos1 - pos2)**2).sum()
                for p, pos1 in enumerate(scene.waypts_pos)
                for pos2 in scene.waypts_pos[p+1:]))
        assert params.catch_area_radius < min_waypts_dist / 2, \
            ("Radius of catchment areas is not less than half of the minimum "
             "distance between waypoints: {}.".format(min_waypts_dist / 2))

        # Validate core region of catchment vector
        assert params.catch_vec_core < 1.0, \
            "Core region of catchment vector is not less than 1."

        # Validate number of routes
        assert len(scene.routes_names) >= 1, "Number of routes is less than 1."

        # Validate routes
        for r, route_waypts_names in enumerate(scene.routes_waypts_names):
            assert route_waypts_names[-1] in scene.goals_names, \
                ("Route {} does not end with one of the goals."
                 "".format(scene.routes_names[r]))

        # Validate data to be saved
        if params.saved_data:
            unprobeable = set(params.saved_data) - set(cls.probeable)
            if unprobeable:
                raise ValueError("The following data is not probeable: {}."
                                 "".format(" ".join(unprobeable)))

    def on_displace(self):
        """Handle notification of displacement."""
        self._displaced = True

    def save_data(self, sim, dirname=None):
        """Save collected data."""

        # If necessary, adjust directory name
        if dirname is None:
            dirname = ""

        # Save static data
        if 'goals_names' in self.params.saved_data:
            np.savetxt(os.path.join(dirname, self.goals_names_filename),
                       self.scene.goals_names, "%s")
        if 'waypts_names' in self.params.saved_data:
            np.savetxt(os.path.join(dirname, self.waypts_names_filename),
                       self.scene.waypts_names, "%s")
        if 'waypts_pos' in self.params.saved_data:
            np.savetxt(os.path.join(dirname, self.waypts_pos_filename),
                       self.scene.waypts_pos, "%.6f")
        if 'routes_waypts_names' in self.params.saved_data:
            data = [[], []]
            for r, route_waypts_names \
                in enumerate(self.scene.routes_waypts_names):
                data[0].extend([r + 1] * len(route_waypts_names))
                data[1].extend(route_waypts_names)
            np.savetxt(
                os.path.join(dirname, self.routes_waypts_names_filename),
                np.array(data).T, "%s")
        if 'routes_waypts_pos' in self.params.saved_data:
            data = [[], []]
            for r, route_waypts_pos in enumerate(self.scene.routes_waypts_pos):
                data[0].extend([[r + 1]] * len(route_waypts_pos))
                data[1].extend(route_waypts_pos)
            np.savetxt(
                os.path.join(dirname, self.routes_waypts_pos_filename),
                np.concatenate(data, axis=1), ["%d", "%.6f", "%.6f"])
        if 'action_names' in self.params.saved_data:
            data = [action.name for action in self.bg.actions.actions]
            np.savetxt(os.path.join(dirname, self.action_names_filename), data,
                       "%s")
        if 'goals_vocab' in self.params.saved_data:
            np.savetxt(os.path.join(dirname, self.goals_vocab_filename),
                       self.goals_vocab.keys, "%s")
        if 'routepts_vocab' in self.params.saved_data:
            np.savetxt(os.path.join(dirname, self.routepts_vocab_filename),
                       self.routepts_vocab.keys, "%s")
        if 'views_vocab' in self.params.saved_data:
            np.savetxt(os.path.join(dirname, self.views_vocab_filename),
                       self.views_vocab.keys, "%s")

        # Save dynamic data
        t_prec = "%.{}f".format(
            len(repr(self.params.nengo_sim_dt).split(".")[-1]))
        t = sim.trange(self.params.vrep_sim_dt)[:,np.newaxis]
        if 'bot_pos' in self.params.saved_data:
            data = sim.data[self.bot_pos_prb]
            np.savetxt(os.path.join(dirname, self.bot_pos_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'bot_orient' in self.params.saved_data:
            data = sim.data[self.bot_orient_prb]
            np.savetxt(os.path.join(dirname, self.bot_orient_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%.4f"])
        if 'global_vec' in self.params.saved_data:
            data = sim.data[self.global_vec_prb]
            np.savetxt(os.path.join(dirname, self.global_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'catch_area' in self.params.saved_data:
            data = sim.data[self.catch_area_prb]
            np.savetxt(os.path.join(dirname, self.catch_area_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%d", "%.6f"])
        if 'catch_vec' in self.params.saved_data:
            data = sim.data[self.catch_vec_prb]
            np.savetxt(os.path.join(dirname, self.catch_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f", "%.6f"])
        if 'vision' in self.params.saved_data:
            data = spa.similarity(sim.data[self.vision_prb], self.views_vocab)
            np.savetxt(os.path.join(dirname, self.vision_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        t = sim.trange()[:,np.newaxis]
        if 'displace' in self.params.saved_data:
            data = sim.data[self.displace_prb]
            np.savetxt(os.path.join(dirname, self.displace_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%d"])
        if 'goal_input' in self.params.saved_data:
            data = spa.similarity(sim.data[self.goal_input_prb][:,1:],
                                  self.goals_vocab)
            np.savetxt(os.path.join(dirname, self.goal_input_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'goal' in self.params.saved_data:
            data = self.similarity(sim.data, self.goal_prb)
            np.savetxt(os.path.join(dirname, self.goal_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'goal_neurons' in self.params.saved_data:
            data = sim.data[self.goal_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.goal_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'goal_detect' in self.params.saved_data:
            data = sim.data[self.goal_detect_prb]
            np.savetxt(os.path.join(dirname, self.goal_detect_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%d"])
        if 'goals2goals_vecs' in self.params.saved_data:
            data = sim.data[self.goals2goals_vecs_prb]
            np.savetxt(os.path.join(dirname, self.goals2goals_vecs_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'target_vec' in self.params.saved_data:
            data = sim.data[self.target_vec_prb]
            np.savetxt(os.path.join(dirname, self.target_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'target_vec_neurons' in self.params.saved_data:
            data = sim.data[self.target_vec_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.target_vec_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'gate_target_vec' in self.params.saved_data:
            data = sim.data[self.gate_target_vec_prb]
            np.savetxt(os.path.join(dirname, self.gate_target_vec_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'gate_target_vec_goal' in self.params.saved_data:
            data = sim.data[self.gate_target_vec_goal_prb]
            np.savetxt(
                os.path.join(dirname, self.gate_target_vec_goal_filename),
                np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'norm_target_vec' in self.params.saved_data:
            data = sim.data[self.norm_target_vec_prb]
            np.savetxt(os.path.join(dirname, self.norm_target_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'view' in self.params.saved_data:
            data = self.similarity(sim.data, self.view_prb)
            np.savetxt(os.path.join(dirname, self.view_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'view_neurons' in self.params.saved_data:
            data = sim.data[self.view_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.view_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'goal0view' in self.params.saved_data:
            data = self.similarity(sim.data, self.goal0view_prb)
            np.savetxt(os.path.join(dirname, self.goal0view_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'view_routept' in self.params.saved_data:
            data = self.similarity(sim.data, self.view_routept_prb)
            np.savetxt(os.path.join(dirname, self.view_routept_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'view_routept_neurons' in self.params.saved_data:
            data = sim.data[self.view_routept_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.view_routept_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'gate_view_routept' in self.params.saved_data:
            data = sim.data[self.gate_view_routept_prb]
            np.savetxt(os.path.join(dirname, self.gate_view_routept_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'prev_routept' in self.params.saved_data:
            data = self.similarity(sim.data, self.prev_routept_prb)
            np.savetxt(os.path.join(dirname, self.prev_routept_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'prev_routept_neurons' in self.params.saved_data:
            data = sim.data[self.prev_routept_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.prev_routept_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'routept_cleanup' in self.params.saved_data:
            data = self.similarity(sim.data, self.routept_cleanup_prb)
            np.savetxt(os.path.join(dirname, self.routept_cleanup_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'prev2next_routepts' in self.params.saved_data:
            data = self.similarity(sim.data, self.prev2next_routepts_prb)
            np.savetxt(os.path.join(dirname, self.prev2next_routepts_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'next_routept' in self.params.saved_data:
            data = self.similarity(sim.data, self.next_routept_prb)
            np.savetxt(os.path.join(dirname, self.next_routept_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'next_routept_neurons' in self.params.saved_data:
            data = sim.data[self.next_routept_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.next_routept_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'gate_routept_cleanup' in self.params.saved_data:
            data = sim.data[self.gate_routept_cleanup_prb]
            np.savetxt(
                os.path.join(dirname, self.gate_routept_cleanup_filename),
                np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'routepts2views' in self.params.saved_data:
            data = self.similarity(sim.data, self.routepts2views_prb)
            np.savetxt(os.path.join(dirname, self.routepts2views_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'view7view' in self.params.saved_data:
            data = sim.data[self.view7view_prb]
            np.savetxt(os.path.join(dirname, self.view7view_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'routepts2norm_local_vecs' in self.params.saved_data:
            data = sim.data[self.routepts2norm_local_vecs_prb]
            np.savetxt(
                os.path.join(dirname, self.routepts2norm_local_vecs_filename),
                np.concatenate((t, data), axis=1), [t_prec, "%.6f", "%.6f"])
        if 'gain_catch_vec' in self.params.saved_data:
            data = sim.data[self.gain_catch_vec_prb]
            np.savetxt(os.path.join(dirname, self.gain_catch_vec_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'gain_catch_vec_neurons' in self.params.saved_data:
            data = sim.data[self.gain_catch_vec_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.gain_catch_vec_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'scaled_catch_vec' in self.params.saved_data:
            data = sim.data[self.scaled_catch_vec_prb]
            np.savetxt(os.path.join(dirname, self.scaled_catch_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'scaled_local_vec' in self.params.saved_data:
            data = sim.data[self.scaled_local_vec_prb]
            np.savetxt(os.path.join(dirname, self.scaled_local_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'scaled_target_vec' in self.params.saved_data:
            data = sim.data[self.scaled_target_vec_prb]
            np.savetxt(os.path.join(dirname, self.scaled_target_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'motion_vec' in self.params.saved_data:
            data = sim.data[self.motion_vec_prb]
            np.savetxt(os.path.join(dirname, self.motion_vec_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'motion_vec_neurons' in self.params.saved_data:
            data = sim.data[self.motion_vec_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.motion_vec_neurons_filename),
                data=np.concatenate((t, data), axis=1))
        if 'wheel_speeds' in self.params.saved_data:
            data = sim.data[self.wheel_speeds_prb]
            np.savetxt(os.path.join(dirname, self.wheel_speeds_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec, "%.6f", "%.6f"])
        if 'gate_reset' in self.params.saved_data:
            data = sim.data[self.gate_reset_prb]
            np.savetxt(os.path.join(dirname, self.gate_reset_filename),
                       np.concatenate((t, data), axis=1), [t_prec, "%.6f"])
        if 'bg' in self.params.saved_data:
            data = sim.data[self.bg_prb]
            np.savetxt(os.path.join(dirname, self.bg_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'thal' in self.params.saved_data:
            data = sim.data[self.thal_prb]
            np.savetxt(os.path.join(dirname, self.thal_filename),
                       np.concatenate((t, data), axis=1),
                       [t_prec] + ["%.6f"] * data.shape[1])
        if 'thal_neurons' in self.params.saved_data:
            data = sim.data[self.thal_neurons_prb]
            np.savez_compressed(
                os.path.join(dirname, self.thal_neurons_filename),
                data=np.concatenate((t, data), axis=1))

    def set_goal(self, goal_name):
        """Set current goal."""
        if goal_name == self._goal_name:  # same goal as already set
            return
        if goal_name not in self.scene.goals_names:
            raise ValueError("Goal '{}' is invalid.".format(goal_name))
        self._new_goal_name = goal_name
        self._goal_reached = False

    @with_self
    def _make_model(self):
        """Create Nengo model."""

        # Determine normalized versions of local vectors between route
        # waypoints
        routes_norm_local_vecs = []
        for route_waypts_pos in self.scene.routes_waypts_pos:
            n_route_waypts = len(route_waypts_pos)
            route_local_vecs = np.zeros((n_route_waypts, 2))
            route_local_vecs[:-1] = np.diff(route_waypts_pos, axis=0)
            route_local_vecs_norms = \
                np.sqrt((route_local_vecs**2).sum(axis=1, keepdims=True))
            route_norm_local_vecs = np.zeros((n_route_waypts, 2))
            route_norm_local_vecs[:-1] = (route_local_vecs[:-1]
                                          / route_local_vecs_norms[:-1])
            routes_norm_local_vecs.append(route_norm_local_vecs)

        # Create vocabularies
        self.goals_vocab = spa.Vocabulary(
            self.params.sp_dim, max_similarity=self.params.sp_max_similarity)
        self.goals_vocab.parse("NONE")
        self.goals_vocab.extend(goal_name.upper()
                                for goal_name in self.scene.goals_names)
        views_sp_keys = ["VIEW_"+waypt_name.upper()
                         for waypt_name in self.scene.waypts_names]
        self.views_vocab = spa.Vocabulary(
            self.params.sp_dim, max_similarity=self.params.sp_max_similarity)
        self.views_vocab.extend(views_sp_keys)
        routes_routepts_sp_keys = []
        self.routepts_vocab = spa.Vocabulary(
            self.params.sp_dim, max_similarity=self.params.sp_max_similarity)
        for route_waypts_names in self.scene.routes_waypts_names:
            route_routepts_sp_keys = []
            goal_sp_key = route_waypts_names[-1].upper()
            goal_sp = self.goals_vocab[goal_sp_key]
            prev_routept_sp = None
            prev_routept_sp_key = None
            for route_waypt_name in route_waypts_names:
                view_sp_key = "VIEW_"+route_waypt_name.upper()
                view_sp = self.views_vocab[view_sp_key]
                routept_sp_key = "{0}x{1}".format(goal_sp_key, view_sp_key)
                route_routepts_sp_keys.append(routept_sp_key)
                try:
                    self.routepts_vocab.add(routept_sp_key, goal_sp * view_sp)
                except nengo.exceptions.ValidationError:
                    pass  # semantic pointer already exists
                routept_sp = self.routepts_vocab[routept_sp_key]
                if prev_routept_sp is not None:
                    similairity = routept_sp.dot(prev_routept_sp)
                    if similairity >= self.params.sp_max_similarity:
                        warnings.warn(
                            "Similarity of semantic pointers {0} and {1} is "
                            "{2}.".format(prev_routept_sp_key, routept_sp_key,
                                          similairity))
                prev_routept_sp = routept_sp
                prev_routept_sp_key = routept_sp_key
            routes_routepts_sp_keys.append(route_routepts_sp_keys)

        # If necessary, create node representing communicator for data exchange
        # with V-REP
        if self.vrep_comm:
            self.vrep_proxy = nengo.Node(self.vrep_comm, size_in=2, size_out=3)

        # Create node representing robot position
        # Outputs (2):
        #  [0] bot_pos_x
        #  [1] bot_pos_y
        if self.vrep_comm:
            self.bot_pos_inp = nengo.Node(None, size_in=2)
            self.vrep_comm.add_output(
                lambda: self.scene.bot.get_position()[:2], 2)
            nengo.Connection(self.vrep_proxy[0:2], self.bot_pos_inp,
                             synapse=None)
        else:
            self.bot_pos_inp = nengo.Node(self.scene.bot.init_pos)

        # Create node representing robot orientation
        # Outputs (1):
        #  [0] bot_orient
        if self.vrep_comm:
            self.bot_orient_inp = nengo.Node(
                lambda t, x: x, size_in=1, size_out=1)
            self.vrep_comm.add_output(
                lambda: self.scene.bot.get_orientation()[2], 1)
            nengo.Connection(self.vrep_proxy[2], self.bot_orient_inp,
                             synapse=None)
        else:
            self.bot_orient_inp = nengo.Node(self.scene.bot.init_orient)

        # Create node representing global vector
        # Inputs (2):
        #  [0] bot_pos_x
        #  [1] bot_pos_y
        # Outputs (2):
        #  [0] global_vec_x
        #  [1] global_vec_y
        self._global_vec = self.params.init_global_vec[:]  # shallow copy to
                                                           # preserve original
                                                           # parameter
        if self.vrep_comm:
            self._bot_prev_pos = self.scene.bot.get_position()[:2]
        else:
            self._bot_prev_pos = self.params.bot_init_pos[:]  # shallow copy

        def update_global_vec(t, x):
            if not self._displaced and self._t_displace is None:
                bot_diff_pos = [x[0] - self._bot_prev_pos[0],
                                x[1] - self._bot_prev_pos[1]]
                if any(bot_diff_pos):
                    self._global_vec[0] += bot_diff_pos[0]
                    self._global_vec[1] += bot_diff_pos[1]
                    self._bot_prev_pos[0] += bot_diff_pos[0]
                    self._bot_prev_pos[1] += bot_diff_pos[1]
            else:
                self._bot_prev_pos = x
            return self._global_vec

        self.global_vec_inp = nengo.Node(update_global_vec, size_in=2,
                                         size_out=2)
        nengo.Connection(self.bot_pos_inp, self.global_vec_inp, synapse=None)

        # Create node representing catchment area
        # Inputs (2):
        #  [0] bot_pos_x
        #  [1] bot_pos_y
        # Outputs (2):
        #  [0] waypt_id
        #  [1] norm_waypt_dist
        def update_catch_area(t, x):
            waypts_square_dists = ((self.scene.waypts_pos - x)**2).sum(axis=1)
            min_waypts_dist = np.sqrt(waypts_square_dists.min())
            if min_waypts_dist < self.params.catch_area_radius:
                norm_waypt_dist = (min_waypts_dist
                                   / self.params.catch_area_radius)
                return [waypts_square_dists.argmin(), norm_waypt_dist]
            return [-1.0, -1.0]

        self.catch_area_inp = nengo.Node(update_catch_area, size_in=2,
                                         size_out=2)
        nengo.Connection(self.bot_pos_inp, self.catch_area_inp, synapse=None)

        # Create node representing catchment vector
        # Inputs (5):
        #  [0] waypt_id
        #  [1] norm_waypt_dist
        #  [2] bot_pos_x
        #  [3] bot_pos_y
        #  [4] view7view
        # Outputs (3):
        #  [0] catch_vec_x
        #  [1] catch_vec_y
        #  [2] catch_vec_gain
        def update_catch_vec(t, x):
            if x[1] > 0.0 and x[4] > self.params.catch_vec_thres:
                catch_vec = self.scene.waypts_pos[int(x[0])] - x[2:4]
                catch_vec_norm = np.sqrt(catch_vec**2).sum()
                catch_vec_gain = np.minimum(
                    (1.0 - x[1]) / (1.0 - self.params.catch_vec_core), 1.0)
                return np.append(catch_vec / catch_vec_norm, catch_vec_gain)
            return [0.0, 0.0, 0.0]

        self.catch_vec_inp = nengo.Node(update_catch_vec, size_in=5,
                                        size_out=3)
        nengo.Connection(self.catch_area_inp, self.catch_vec_inp[0:2],
                         synapse=None)
        nengo.Connection(self.bot_pos_inp, self.catch_vec_inp[2:4],
                         synapse=None)

        # Create node representing vision
        # Inputs (2):
        #  [0] waypt_id
        #  [1] norm_waypt_dist
        # Outputs (1):
        #  [0:sp_dim] view
        def update_vision(t, x):
            if x[1] >= 0.0:
                gain = 1.0 - 0.7 * x[1]
                return gain * self.views_vocab[views_sp_keys[int(x[0])]].v
            else:
                return [0.0] * self.params.sp_dim

        self.vision_inp = nengo.Node(update_vision, size_in=2,
                                     size_out=self.params.sp_dim)
        nengo.Connection(self.catch_area_inp, self.vision_inp, synapse=None)

        # Create node representing displacement
        # Outputs (1):
        #  [0] displace_detected
        self._t_displace = None

        def update_displace(t):
            if self._displaced:
                self._t_displace = t
                self._displaced = False
                return True
            if self._t_displace is not None:
                if t < self._t_displace + 0.1:
                    return True
                else:
                    self._t_displace = None
            return False

        self.displace_inp = nengo.Node(update_displace, size_in=0, size_out=1)

        # Create node representing new goal
        # Inputs (1):
        #  [0:sp_dim] goal (current)
        # Outputs (2):
        #  [0] goal_set
        #  [1:sp_dim+1] goal (difference between new and current)
        self._t_new_goal = None
        self._goal_sp = None

        def update_goal_input(t, x):
            if self._new_goal_name is not None:
                self._goal_name = self._new_goal_name
                self._t_new_goal = t
                self._goal_sp = self.goals_vocab[self._goal_name.upper()].v - x
                self._new_goal_name = None
                return np.append([True], self._goal_sp)
            if self._t_new_goal is not None:
                if t < self._t_new_goal + 0.1:
                    return np.append([True], self._goal_sp)
                else:
                    self._t_new_goal = None
                    self._goal_sp = None
            return np.append([False], [0.0] * self.params.sp_dim)

        self.goal_input = nengo.Node(update_goal_input,
                                     size_in=self.params.sp_dim,
                                     size_out=1+self.params.sp_dim)

        # Create state representing current goal
        self.goal = spa.State(self.params.sp_dim, vocab=self.goals_vocab,
                              feedback=1)
        nengo.Connection(self.goal_input[1:], self.goal.input)
        nengo.Connection(self.goal.output, self.goal_input, synapse=0.1)

        # Create node representing goal detection
        self._t_goal_reached = None

        # Inputs (2):
        #  [0:sp_dim] goal
        #  [sp_dim:2*sp_dim] prev_routept
        # Outputs (1):
        #  [0] goal_reached
        def update_goal_detect(t, x):
            sp_dim = self.params.sp_dim
            goals_dots = spa.similarity(x[0:sp_dim], self.goals_vocab)
            if goals_dots.max() < 0.45:  # insufficient similarity to any goal
                                         # (possibly no current goal set)
                self._t_goal_reached = None
                return self._goal_reached
            routepts_dots = spa.similarity(x[sp_dim:2*sp_dim],
                                           self.routepts_vocab)
            if routepts_dots.max() < 0.45:  # insufficient similarity to any
                                            # routepoint (possibly no previous
                                            # routepoint reached)
                self._t_goal_reached = None
                return self._goal_reached
            g = goals_dots.argmax()
            if g == 0:  # "NONE"
                self._t_goal_reached = None
                return self._goal_reached
            goal_name = self.scene.goals_names[g-1]
            routept_sp_key = "{0}xVIEW_{0}".format(goal_name.upper())
            r = routepts_dots.argmax()
            if self.routepts_vocab.keys[r] == routept_sp_key:
                if self._t_goal_reached is None:
                    self._t_goal_reached = t
                else:
                    if t > self._t_goal_reached + 0.2:
                        self._t_goal_reached = None
                        self._goal_reached = True
            else:
                self._t_goal_reached = None
            return self._goal_reached

        self.goal_detect_inp = nengo.Node(
            update_goal_detect, size_in=2*self.params.sp_dim, size_out=1)
        nengo.Connection(self.goal.output,
                         self.goal_detect_inp[0:self.params.sp_dim])

        # Create associative memory between goals and goal vectors
        self.goals2goals_vecs = networks.AssociativeMemory(
            input_vectors=self.goals_vocab.vectors[1:],
            output_vectors=self.scene.goals_pos)
        nengo.Connection(self.goal.output, self.goals2goals_vecs.input)

        # Create ensemble representing target vector
        self.target_vec = nengo.Ensemble(int(10.0 * self.params.scene_radius),
                                         dimensions=2,
                                         radius=self.params.scene_radius)
        nengo.Connection(self.goals2goals_vecs.output, self.target_vec)
        nengo.Connection(self.global_vec_inp, self.target_vec, transform=-1.0)

        # Create gate for inhibiting target vector
        with presets.ThresholdingEnsembles(0.3):
            self.gate_target_vec = spa.State(1, feedback=1,
                                             feedback_synapse=0.02)
        nengo.Connection(self.gate_target_vec.output, self.target_vec.neurons,
                         transform=[[-5.0]]*self.target_vec.n_neurons)

        # Create gate for inhibiting target vector depending on whether current
        # goal is set
        with presets.ThresholdingEnsembles(0.3):
            self.gate_target_vec_goal = spa.State(1)
        nengo.Connection(self.gate_target_vec_goal.output,
                         self.target_vec.neurons,
                         transform=[[-5.0]]*self.target_vec.n_neurons)

        # Create ensemble representing normalized target vector
        def normalize_target_vec(x):
            if any(x):
                return x / np.sqrt((x**2).sum())
            else:
                return [0.0, 0.0]

        self.norm_target_vec = nengo.Ensemble(200, dimensions=2)
        nengo.Connection(self.target_vec, self.norm_target_vec,
                         function=normalize_target_vec)

        # Create state representing current view
        self.view = spa.State(self.params.sp_dim, vocab=self.views_vocab)
        nengo.Connection(self.vision_inp, self.view.input)

        # Create binder for current goal and current view
        self.goal0view = spa.Bind(self.params.sp_dim,
                                  vocab=self.routepts_vocab)
        nengo.Connection(self.goal.output, self.goal0view.A)
        nengo.Connection(self.view.output, self.goal0view.B)

        # Create state representing currently viewed routepoint
        self.view_routept = spa.State(self.params.sp_dim,
                                      vocab=self.routepts_vocab)
        self.view_routept.state_ensembles.add_neuron_input()
        nengo.Connection(self.goal0view.output, self.view_routept.input)

        # Create gate for inhibiting currently viewed routepoint
        with presets.ThresholdingEnsembles(0.3):
            self.gate_view_routept = spa.State(1, feedback=1,
                                               feedback_synapse=0.02)
        nengo.Connection(
            self.gate_view_routept.output,
            self.view_routept.state_ensembles.neuron_input,
            transform=[[-5.0]]*self.view_routept.state_ensembles.n_neurons)

        # Create state representing previous routepoint
        self.prev_routept = spa.State(self.params.sp_dim,
                                      vocab=self.routepts_vocab)
        nengo.Connection(
            self.prev_routept.output,
            self.goal_detect_inp[self.params.sp_dim:2*self.params.sp_dim])

        # Create cleanup memory for routepoints
        self.routept_cleanup = spa.AssociativeMemory(
            input_vocab=self.routepts_vocab, inhibitable=True, wta_output=True)
        nengo.Connection(self.prev_routept.output, self.routept_cleanup.input)
        nengo.Connection(self.routept_cleanup.output, self.prev_routept.input)

        # Create associative memory between previous and next routepoints
        input_keys = [routept_sp_key
                      for route_routepts_sp_keys in routes_routepts_sp_keys
                      for routept_sp_key in route_routepts_sp_keys[:-1]]
        output_keys = [routept_sp_key
                       for route_routepts_sp_keys in routes_routepts_sp_keys
                       for routept_sp_key in route_routepts_sp_keys[1:]]
        self.prev2next_routepts = spa.AssociativeMemory(
            input_vocab=self.routepts_vocab, output_vocab=self.routepts_vocab,
            input_keys=input_keys, output_keys=output_keys, wta_output=True)
        nengo.Connection(self.prev_routept.output,
                         self.prev2next_routepts.input)

        # Create state representing next routepoint
        self.next_routept = spa.State(self.params.sp_dim,
                                      vocab=self.routepts_vocab)
        nengo.Connection(self.prev2next_routepts.output,
                         self.next_routept.input)

        # Create gate for inhibiting cleanup memory for routepoints
        self.gate_routept_cleanup = spa.State(1)

        # Create network transiently inhibiting cleanup memory for routepoints
        self.routept_cleanup_inhibit = nengo.Network()
        with self.routept_cleanup_inhibit as net:
            # Create (input) excitatory ensemble
            net.ens_exc = nengo.Ensemble(100, dimensions=1)

            # Create excitatory ensemble in the internal loop
            net.ens_exc_loop = nengo.Ensemble(100, dimensions=1)
            nengo.Connection(net.ens_exc, net.ens_exc_loop, synapse=0.02)

            # Create inhibitory ensemble in the internal loop
            net.ens_inh_loop = nengo.Ensemble(100, dimensions=1)
            nengo.Connection(net.ens_exc_loop, net.ens_inh_loop)
            nengo.Connection(net.ens_inh_loop, net.ens_exc.neurons,
                             transform=[[-20.0]]*net.ens_exc.n_neurons,
                             synapse=0.2)

            # Create (output) inhibitory ensemble
            with presets.ThresholdingEnsembles(0.1):
                net.ens_inh = nengo.Ensemble(100, dimensions=1)
            nengo.Connection(net.ens_exc, net.ens_inh, synapse=0.1)

        nengo.Connection(self.gate_routept_cleanup.output,
                         self.routept_cleanup_inhibit.ens_exc)
        nengo.Connection(self.routept_cleanup_inhibit.ens_inh,
                         self.routept_cleanup.inhibit, transform=6.0)

        # Create associative memory between routepoints and views
        input_keys = [routept_sp_key
                      for route_routepts_sp_keys in routes_routepts_sp_keys
                      for routept_sp_key in route_routepts_sp_keys]
        output_keys = ["VIEW_"+route_waypt_name.upper()
                       for route_waypts_names in self.scene.routes_waypts_names
                       for route_waypt_name in route_waypts_names]
        self.routepts2views = spa.AssociativeMemory(
            input_vocab=self.routepts_vocab, output_vocab=self.views_vocab,
            input_keys=input_keys, output_keys=output_keys, wta_output=True)
        nengo.Connection(self.prev2next_routepts.output,
                         self.routepts2views.input)
        nengo.Connection(self.view_routept.output, self.routepts2views.input)

        # Create comparator between current and retrieved views
        self.view7view = spa.Compare(self.params.sp_dim,
                                     vocab=self.views_vocab)
        nengo.Connection(self.view.output, self.view7view.inputA)
        nengo.Connection(self.routepts2views.output, self.view7view.inputB)
        nengo.Connection(self.view7view.output, self.catch_vec_inp[4],
                         synapse=0.02)

        # Create associative memory between routepoints and normalized local
        # vectors between route waypoints
        input_vectors = [self.routepts_vocab[routept_sp_key].v
                         for route_routepts_sp_keys in routes_routepts_sp_keys
                         for routept_sp_key in route_routepts_sp_keys]
        output_vectors = np.concatenate(routes_norm_local_vecs)
        self.routepts2norm_local_vecs = networks.AssociativeMemory(
            input_vectors=input_vectors, output_vectors=output_vectors)
        self.routepts2norm_local_vecs.add_wta_network(inhibit_scale=3.0)
        nengo.Connection(self.prev_routept.output,
                         self.routepts2norm_local_vecs.input)

        # Create ensemble representing gain of the catchment vector
        self.gain_catch_vec = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(self.catch_vec_inp[2], self.gain_catch_vec,
                         synapse=None)

        # Create network representing scaled catchment vector
        self.scaled_catch_vec = networks.Product(200, dimensions=2,
                                                 label="scaled_catch_vec")
        nengo.Connection(self.catch_vec_inp[0:2], self.scaled_catch_vec.A,
                         synapse=None)
        nengo.Connection(self.gain_catch_vec, self.scaled_catch_vec.B,
                         transform=[[1.0], [1.0]])

        # Create network representing scaled local vector
        self.scaled_local_vec = networks.Product(200, dimensions=2,
                                                 label="scaled_local_vec")
        nengo.Connection(self.routepts2norm_local_vecs.output,
                         self.scaled_local_vec.A)
        nengo.Connection(self.gain_catch_vec, self.scaled_local_vec.B,
                         function=lambda x: 1.0 - x, transform=[[1.0], [1.0]])

        # Create network representing scaled target vector
        self.scaled_target_vec = networks.Product(200, dimensions=2,
                                                  label="scaled_target_vec")
        nengo.Connection(self.norm_target_vec, self.scaled_target_vec.A)
        nengo.Connection(self.gain_catch_vec, self.scaled_target_vec.B,
                         function=lambda x: 1.0 - x, transform=[[1.0], [1.0]])

        # Create network aggregating scaled vectors and transforming them to
        # the robot reference frame
        self.motion_vec_aggreg = networks.Product(200, dimensions=4,
                                                  label="motion_vec_aggreg")
        nengo.Connection(
            self.scaled_catch_vec.output, self.motion_vec_aggreg.A,
            transform=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        nengo.Connection(
            self.scaled_local_vec.output, self.motion_vec_aggreg.A,
            transform=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        nengo.Connection(
            self.scaled_target_vec.output, self.motion_vec_aggreg.A,
            transform=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        nengo.Connection(
            self.bot_orient_inp, self.motion_vec_aggreg.B,
            function=lambda x: (np.cos(x), np.sin(x)),
            transform=[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]])

        # Create ensemble representing motion vector
        self.motion_vec = nengo.Ensemble(100, dimensions=2)
        nengo.Connection(
            self.motion_vec_aggreg.output, self.motion_vec,
            transform=[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

        # Create node representing wheel speeds
        # Inputs (2):
        #  [0] motion_vec_x
        #  [1] motion_vec_y
        # Outputs (2):
        #  [0] left_wheel_speed
        #  [1] right_wheel_speed
        def update_wheel_speeds(t, x):
            if np.sqrt((x**2).sum()) < 0.05:
                return [0.0, 0.0]  # stop
            theta = np.arctan2(x[1], x[0])
            if -0.4 < theta < 0.4:
                return [5.0-theta, 5.0+theta]  # (curvilinear) motion forward
            else:
                return [-5.0*theta, 5.0*theta]  # rotation on the spot

        self.wheel_speeds = nengo.Node(update_wheel_speeds, size_in=2,
                                       size_out=2)
        nengo.Connection(self.motion_vec, self.wheel_speeds)
        if self.vrep_comm:
            self.vrep_comm.add_input(self.scene.bot.wheels.set_velocities, 2)
            nengo.Connection(self.wheel_speeds, self.vrep_proxy[0:2],
                             synapse=None)

        # Create gate for reset signal
        self.gate_reset = spa.State(1)
        nengo.Connection(self.displace_inp, self.gate_reset.input)
        nengo.Connection(self.goal_input[0], self.gate_reset.input)

        # Create action selection circuit
        actions = spa.Actions(
            reset=\
                ('gate_reset -->'
                 ' gate_view_routept = -1.0,'
                 ' gate_target_vec = -1.0,'
                 ' gate_routept_cleanup = 1.0'),
            no_goal=\
                'dot(goal, NONE) --> gate_target_vec_goal = 1.0',
            next2prev_routept=\
                ('0.5 * view7view + {} * gate_view_routept -->'
                 ' prev_routept = next_routept,'
                 ' next_routept = next_routept,'
                 ' gate_routept_cleanup = 1.0'
                 ''.format(self.params.action_n2pr_coeff_gvr)),
            view2prev_routept=\
                ('0.5 * view7view - {0} * gate_view_routept + {1} -->'
                 ' prev_routept = 3.0 * goal0view,'
                 ' gate_view_routept = 1.0,'
                 ' gate_target_vec = 1.0'
                 ''.format(self.params.action_v2pr_coeff_gvr,
                           self.params.action_v2pr_coeff_intercept)),
            default=\
                '0.5 -->')
        self.bg = spa.BasalGanglia(actions)
        self.thal = spa.Thalamus(self.bg)

    @with_self
    def _make_probes(self):
        """Create probes for recording dynamic data."""

        if 'bot_pos' in self.params.saved_data:
            self.bot_pos_prb = nengo.Probe(
                self.bot_pos_inp, sample_every=self.params.vrep_sim_dt)
        if 'bot_orient' in self.params.saved_data:
            self.bot_orient_prb = nengo.Probe(
                self.bot_orient_inp, sample_every=self.params.vrep_sim_dt)
        if 'global_vec' in self.params.saved_data:
            self.global_vec_prb = nengo.Probe(
                self.global_vec_inp, sample_every=self.params.vrep_sim_dt)
        if 'catch_area' in self.params.saved_data:
            self.catch_area_prb = nengo.Probe(
                self.catch_area_inp, sample_every=self.params.vrep_sim_dt)
        if 'catch_vec' in self.params.saved_data:
            self.catch_vec_prb = nengo.Probe(
                self.catch_vec_inp, sample_every=self.params.vrep_sim_dt)
        if 'vision' in self.params.saved_data:
            self.vision_prb = nengo.Probe(self.vision_inp,
                                          sample_every=self.params.vrep_sim_dt)
        if 'displace' in self.params.saved_data:
            self.displace_prb = nengo.Probe(self.displace_inp)
        if 'goal_input' in self.params.saved_data:
            self.goal_input_prb = nengo.Probe(self.goal_input)
        if 'goal' in self.params.saved_data:
            self.goal_prb = nengo.Probe(self.goal.output,
                                        synapse=self.params.prb_syn)
        if 'goal_neurons' in self.params.saved_data:
            self.goal_neurons_prb = nengo.Probe(
                self.goal.state_ensembles.add_neuron_output())
        if 'goal_detect' in self.params.saved_data:
            self.goal_detect_prb = nengo.Probe(self.goal_detect_inp)
        if 'goals2goals_vecs' in self.params.saved_data:
            self.goals2goals_vecs_prb = nengo.Probe(
                self.goals2goals_vecs.output, synapse=self.params.prb_syn)
        if 'target_vec' in self.params.saved_data:
            self.target_vec_prb = nengo.Probe(self.target_vec,
                                              synapse=self.params.prb_syn)
        if 'target_vec_neurons' in self.params.saved_data:
            self.target_vec_neurons_prb = nengo.Probe(self.target_vec.neurons)
        if 'gate_target_vec' in self.params.saved_data:
            self.gate_target_vec_prb = nengo.Probe(self.gate_target_vec.output,
                                                   synapse=self.params.prb_syn)
        if 'gate_target_vec_goal' in self.params.saved_data:
            self.gate_target_vec_goal_prb = nengo.Probe(
                self.gate_target_vec_goal.output, synapse=self.params.prb_syn)
        if 'norm_target_vec' in self.params.saved_data:
            self.norm_target_vec_prb = nengo.Probe(self.norm_target_vec,
                                                   synapse=self.params.prb_syn)
        if 'view' in self.params.saved_data:
            self.view_prb = nengo.Probe(self.view.output,
                                        synapse=self.params.prb_syn)
        if 'view_neurons' in self.params.saved_data:
            self.view_neurons_prb = nengo.Probe(
                self.view.state_ensembles.add_neuron_output())
        if 'goal0view' in self.params.saved_data:
            self.goal0view_prb = nengo.Probe(self.goal0view.output,
                                             synapse=self.params.prb_syn)
        if 'view_routept' in self.params.saved_data:
            self.view_routept_prb = nengo.Probe(self.view_routept.output,
                                                synapse=self.params.prb_syn)
        if 'view_routept_neurons' in self.params.saved_data:
            self.view_routept_neurons_prb = nengo.Probe(
                self.view_routept.state_ensembles.add_neuron_output())
        if 'gate_view_routept' in self.params.saved_data:
            self.gate_view_routept_prb = nengo.Probe(
                self.gate_view_routept.output, synapse=self.params.prb_syn)
        if 'prev_routept' in self.params.saved_data:
            self.prev_routept_prb = nengo.Probe(self.prev_routept.output,
                                                synapse=self.params.prb_syn)
        if 'prev_routept_neurons' in self.params.saved_data:
            self.prev_routept_neurons_prb = nengo.Probe(
                self.prev_routept.state_ensembles.add_neuron_output())
        if 'routept_cleanup' in self.params.saved_data:
            self.routept_cleanup_prb = nengo.Probe(self.routept_cleanup.output,
                                                   synapse=self.params.prb_syn)
        if 'prev2next_routepts' in self.params.saved_data:
            self.prev2next_routepts_prb = nengo.Probe(
                self.prev2next_routepts.output, synapse=self.params.prb_syn)
        if 'next_routept' in self.params.saved_data:
            self.next_routept_prb = nengo.Probe(self.next_routept.output,
                                                synapse=self.params.prb_syn)
        if 'next_routept_neurons' in self.params.saved_data:
            self.next_routept_neurons_prb = nengo.Probe(
                self.next_routept.state_ensembles.add_neuron_output())
        if 'gate_routept_cleanup' in self.params.saved_data:
            self.gate_routept_cleanup_prb = nengo.Probe(
                self.gate_routept_cleanup.output, synapse=self.params.prb_syn)
        if 'routepts2views' in self.params.saved_data:
            self.routepts2views_prb = nengo.Probe(self.routepts2views.output,
                                                  synapse=self.params.prb_syn)
        if 'view7view' in self.params.saved_data:
            self.view7view_prb = nengo.Probe(self.view7view.output,
                                             synapse=self.params.prb_syn)
        if 'routepts2norm_local_vecs' in self.params.saved_data:
            self.routepts2norm_local_vecs_prb = nengo.Probe(
                self.routepts2norm_local_vecs.output,
                synapse=self.params.prb_syn)
        if 'gain_catch_vec' in self.params.saved_data:
            self.gain_catch_vec_prb = nengo.Probe(self.gain_catch_vec,
                                                  synapse=self.params.prb_syn)
        if 'gain_catch_vec_neurons' in self.params.saved_data:
            self.gain_catch_vec_neurons_prb = nengo.Probe(
                self.gain_catch_vec.neurons)
        if 'scaled_catch_vec' in self.params.saved_data:
            self.scaled_catch_vec_prb = nengo.Probe(
                self.scaled_catch_vec.output, synapse=self.params.prb_syn)
        if 'scaled_local_vec' in self.params.saved_data:
            self.scaled_local_vec_prb = nengo.Probe(
                self.scaled_local_vec.output, synapse=self.params.prb_syn)
        if 'scaled_target_vec' in self.params.saved_data:
            self.scaled_target_vec_prb = nengo.Probe(
                self.scaled_target_vec.output, synapse=self.params.prb_syn)
        if 'motion_vec' in self.params.saved_data:
            self.motion_vec_prb = nengo.Probe(self.motion_vec,
                                              synapse=self.params.prb_syn)
        if 'motion_vec_neurons' in self.params.saved_data:
            self.motion_vec_neurons_prb = nengo.Probe(self.motion_vec.neurons)
        if 'wheel_speeds' in self.params.saved_data:
            self.wheel_speeds_prb = nengo.Probe(self.wheel_speeds)
        if 'gate_reset' in self.params.saved_data:
            self.gate_reset_prb = nengo.Probe(self.gate_reset.output,
                                              synapse=self.params.prb_syn)
        if 'bg' in self.params.saved_data:
            self.bg_prb = nengo.Probe(self.bg.input,
                                      synapse=self.params.prb_syn)
        if 'thal' in self.params.saved_data:
            self.thal_prb = nengo.Probe(self.thal.actions.output,
                                        synapse=self.params.prb_syn)
        if 'thal_neurons' in self.params.saved_data:
            self.thal_neurons_prb = nengo.Probe(
                self.thal.actions.add_neuron_output())
