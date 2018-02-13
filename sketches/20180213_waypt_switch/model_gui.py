"""Switching between waypoints along a route.

This script provides a Nengo model that imitates using views to proceed from
one waypoint to the next along a route. Views are mimicked in such a way that
there is a unique view related to each waypoint and it is represented as a
semantic pointer of arbitrary numbers. Upon entering the catchment area of a
waypoint, the semantic pointer representing the corresponding view is retrieved
and scaled depending on the remaining distance to the waypoint. Waypoints are
also represented as semantic pointers, and each waypoint is associated with its
successor as well as with the corresponding view. Moreover, information about
the previous waypoint is maintained in working memory, which consists of two
reciprocally connected components: a state that expresses the semantic pointer
representing the previous waypoint and cleanup memory for waypoints. When the
robot approaches the next waypoint, the working memory needs to be updated,
namely the next waypoint should become a new previous waypoint. Such update is
mediated by the action selection circuit, which initiates it once it detects
that similarity between the semantic pointer representing the current view and
the semantic pointer representing the view at the next waypoint, retrieved from
the relevant associative memory, is high. If this is the case, the action
selection circuit feeds the semantic pointer representing the next waypoint
into the state that expresses the semantic pointer representing the previous
waypoint and also triggers transient inhibition of the cleanup memory for
waypoints to prevent the latter from interfering during this process. This
transient inhibition is mediated by an inhibitory ensemble, which is
interconnected with other excitatory and inhibitory ensembles in such a way
that the resulting dynamics makes the transient period long enough for the
update process to complete. Additionally, the action selection circuit
activates an excitatory feedback loop from the state that expresses the
semantic pointer representing the next waypoint to itself, which is necessary
because the input to this state undergoes transition and the update process
could fail otherwise. Following a route is imitated by advancing the robot
position, which must be manually changed using Nengo GUI.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import nengo.spa as spa
import numpy as np
import simtools

waypts_names = ["Nest", "Waypoint", "Feed"]

# --- PARAMETERS ---
class Params(object): pass
params = Params()

# Nengo model parameters
params.sp_dim = 64

# Random number generator parameters
params.np_seed = None
# ------------------

# Set random seed
if params.np_seed is None:
    params.np_seed = simtools.generate_seed(4)
np.random.seed(params.np_seed)
print("NumPy seed: {}".format(params.np_seed))

# Create vocabularies
views_sp_keys = ["VIEW_"+waypt_name.upper() for waypt_name in waypts_names]
views_vocab = spa.Vocabulary(params.sp_dim)
views_vocab.extend(views_sp_keys)
waypts_sp_keys = [waypt_name.upper() for waypt_name in waypts_names]
waypts_vocab = spa.Vocabulary(params.sp_dim)
waypts_vocab.extend(waypts_sp_keys)

# Create Nengo model controlling the robot
model = spa.SPA()
with model:
    # Create node representing robot position
    bot_pos_inp = nengo.Node(0)

    # Create node representing vision
    def update_vision(t, x):
        waypt_idx = int(round(x))
        scale = 1.0 - (abs(x - waypt_idx) / 0.5)
        return scale * views_vocab[views_sp_keys[waypt_idx]].v

    vision_inp = nengo.Node(update_vision, size_in=1, size_out=params.sp_dim)
    nengo.Connection(bot_pos_inp, vision_inp, synapse=None)

    # Create state representing current view
    model.view = spa.State(params.sp_dim, vocab=views_vocab)
    nengo.Connection(vision_inp, model.view.input)

    # Create state representing previous waypoint
    model.prev_waypt = spa.State(params.sp_dim, vocab=waypts_vocab)

    # Create cleanup memory for waypoints
    model.waypt_cleanup = spa.AssociativeMemory(
        input_vocab=waypts_vocab, inhibitable=True, wta_output=True)
    nengo.Connection(model.prev_waypt.output, model.waypt_cleanup.input)
    nengo.Connection(model.waypt_cleanup.output, model.prev_waypt.input)

    # Create associative memory between previous and next waypoints
    model.prev2next_waypts = spa.AssociativeMemory(
        input_vocab=waypts_vocab, output_vocab=waypts_vocab,
        input_keys=waypts_sp_keys[:-1], output_keys=waypts_sp_keys[1:],
        wta_output=True)
    nengo.Connection(model.prev_waypt.output, model.prev2next_waypts.input)

    # Create state representing next waypoint
    model.next_waypt = spa.State(params.sp_dim, vocab=waypts_vocab)
    nengo.Connection(model.prev2next_waypts.output, model.next_waypt.input)

    # Create gate for inhibiting cleanup memory for waypoints
    model.gate_inh_waypt_cleanup = spa.State(1)

    # Create intermediate excitatory ensemble
    ens_exc = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(model.gate_inh_waypt_cleanup.output, ens_exc)

    # Create excitatory ensemble in the internal loop
    ens_exc_loop = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(ens_exc, ens_exc_loop, synapse=0.02)

    # Create inhibitory ensemble in the internal loop
    ens_inh_loop = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(ens_exc_loop, ens_inh_loop)
    nengo.Connection(ens_inh_loop, ens_exc.neurons, transform=[[-20.0]]*100,
                     synapse=0.2)

    # Create inhibitory ensemble
    ens_inh = nengo.Ensemble(
        100, dimensions=1, encoders=nengo.dists.Choice([[1]]),
        intercepts=nengo.dists.Exponential(0.15, 0.1, 1.0))
    nengo.Connection(ens_exc, ens_inh, synapse=0.1)
    nengo.Connection(ens_inh, model.waypt_cleanup.inhibit, transform=6.0)

    # Create associative memory between waypoints and views
    model.waypts2views = spa.AssociativeMemory(
        input_vocab=waypts_vocab, output_vocab=views_vocab,
        input_keys=waypts_sp_keys, output_keys=views_sp_keys, wta_output=True)
    nengo.Connection(model.prev2next_waypts.output, model.waypts2views.input)

    # Create comparator between current and next views
    model.view7view = spa.Compare(params.sp_dim, vocab=views_vocab)
    nengo.Connection(model.view.output, model.view7view.inputA)
    nengo.Connection(model.waypts2views.output, model.view7view.inputB)

    # Create action selection circuit
    actions = spa.Actions(
        ('view7view -->'
         '    prev_waypt = next_waypt,'
         '    next_waypt = next_waypt,'
         '    gate_inh_waypt_cleanup = 1'),
        '0.5 -->')
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    # Create input initializing previous waypoint
    model.prev_waypt_init = spa.Input(
        prev_waypt=lambda t: "NEST" if t < 0.1 else '0')
