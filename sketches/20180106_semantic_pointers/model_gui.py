"""Adaptation of semantic pointers for navigation and view representation.

This script provides a Nengo model that uses semantic pointers to represent
views and other information regarding navigation when following routes. The
first group of semantic pointers represents views. Views are mimicked in such a
way that there is a unique view related to each waypoint and it is represented
as a semantic pointer of arbitrary numbers. Upon entering the catchment area of
a waypoint, the semantic pointer representing the corresponding view is
retrieved and scaled depending on the remaining distance to the waypoint. The
second group of semantic pointers represents goals, which are final waypoints
of the outbound and inbound routes. Finally, every semantic pointer in the
third group forms a compositional representation that directly binds a
particular goal to a particular view. Both the semantic pointer representing
the goal and the semantic pointer representing the view can be independently
retrieved from the compositional representation. When a route is followed,
traversing catchment areas of consecutive waypoints results in retrieval of
semantic pointers representing the corresponding views, which are scaled
according to the distance to the waypoint and then bound to the semantic
pointer representing the current goal to yield a compositional representation.
Following a route is imitated by advancing the robot position, which must be
manually changed using Nengo GUI. Likewise, the current goal can also be
manually changed after initialization using Nengo GUI.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import nengo.spa as spa
import numpy as np

waypts_pos = [0.0, 2.0, 4.0, 6.0]
waypts_names = ["NEST", "WAYPOINT1", "WAYPOINT2", "FEED"]
goals_names = ["NEST", "FEED"]

# --- PARAMETERS ---
class Params(object): pass
params = Params()

# Location parameters
params.catch_area_radius = 0.5

# Nengo model parameters
params.sp_dim = 32
params.unitary_sps = True
# ------------------

# Validate radius of catchment areas
min_waypts_dist = (np.diff(waypts_pos)).min()
assert min_waypts_dist > params.catch_area_radius * 2, \
    ("Radius of catchment areas must be less than {}."
     "".format(min_waypts_dist / 2))

# Create vocabularies
views_names = ["VIEW_"+waypt_name for waypt_name in waypts_names]
views_vocab = spa.Vocabulary(params.sp_dim, unitary=params.unitary_sps)
views_vocab.extend(views_names)
goals_vocab = spa.Vocabulary(params.sp_dim, unitary=params.unitary_sps)
goals_vocab.extend(goals_names)
routepts_vocab = spa.Vocabulary(params.sp_dim)
for goal_name in goals_names:
    goal_sp = goals_vocab[goal_name]
    for view_name in views_names:
        view_sp = views_vocab[view_name]
        routepts_vocab.add("{0}*{1}".format(goal_name, view_name),
                           goal_sp * view_sp)

# Create Nengo model
model = spa.SPA()
with model:
    # Create node representing robot position
    model.bot_pos_inp = nengo.Node([0], size_out=1)

    # Create node representing current view
    def update_view(t, x):
        waypts_dists = np.sqrt(((x - waypts_pos)**2))
        min_waypt_dist = waypts_dists.min()
        if min_waypt_dist < params.catch_area_radius:
            view_name = views_names[waypts_dists.argmin()]
            gain = 1.0 - (min_waypt_dist / params.catch_area_radius)
            return gain * views_vocab[view_name].v
        else:
            return [0.0] * params.sp_dim

    model.view_inp = nengo.Node(update_view, size_in=1, size_out=params.sp_dim)
    nengo.Connection(model.bot_pos_inp, model.view_inp, synapse=None)

    # Create state representing current view
    model.view = spa.State(params.sp_dim, vocab=views_vocab)
    nengo.Connection(model.view_inp, model.view.input)

    # Create state representing current goal
    model.goal = spa.State(params.sp_dim, vocab=goals_vocab, feedback=1)

    # Create binder representing compositional representation of the current
    # goal and the current view
    model.bind = spa.Bind(params.sp_dim, vocab=routepts_vocab)
    nengo.Connection(model.goal.output, model.bind.A)
    nengo.Connection(model.view.output, model.bind.B)

    # Create binder representing current goal retrieved from the compositional
    # representation
    model.goal_estim = spa.Bind(params.sp_dim, vocab=goals_vocab,
                                invert_b=True)
    nengo.Connection(model.bind.output, model.goal_estim.A)
    nengo.Connection(model.view.output, model.goal_estim.B)

    # Create binder representing current view retrieved from the compositional
    # representation
    model.view_estim = spa.Bind(params.sp_dim, vocab=views_vocab,
                                invert_b=True)
    nengo.Connection(model.bind.output, model.view_estim.A)
    nengo.Connection(model.goal.output, model.view_estim.B)

    # Create input initializing current goal
    model.goal_init = spa.Input(goal=lambda t: "FEED" if t < 0.1 else '0')
