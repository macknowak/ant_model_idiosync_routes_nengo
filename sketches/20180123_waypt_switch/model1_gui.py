"""Switching between waypoints along a route.

This script provides a Nengo model that maintains information about the
previous waypoint in working memory when following a route. Waypoints are
represented as semantic pointers. The working memory consists of two
components, which are reciprocally connected to each other: the first component
is a state that expresses the semantic pointer representing the previous
waypoint, whereas the second component is autoassociative memory that stores
semantic pointers representing all waypoints and thus acts as cleanup memory.
When the robot approaches the next waypoint, the working memory needs to be
updated, namely the next waypoint should become a new previous waypoint. To
this end, the difference between the semantic pointer representing the next
waypoint and the semantic pointer representing the previous waypoint is
determined and fed into the state that expresses the semantic pointer
representing the previous waypoint. Following a route is imitated by advancing
the robot position, which must be manually changed using Nengo GUI.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import nengo.spa as spa
import numpy as np

waypts_names = ["NEST", "WAYPOINT1", "FEED"]
waypts_pos = [0.0, 1.0, 2.0]

# --- PARAMETERS ---
class Params(object): pass
params = Params()

# Nengo model parameters
params.sp_dim = 64
# ------------------

# Create vocabulary
waypts_vocab = spa.Vocabulary(params.sp_dim)
waypts_vocab.extend(waypts_names)

# Create Nengo model
model = spa.SPA()
with model:
    # Create node representing robot position
    bot_pos_inp = nengo.Node(0.0, size_out=1)

    # Create node representing next waypoint
    def update_next_waypt(t, x):
        return waypts_vocab[waypts_names[int(np.floor(x + 0.5))]].v

    next_waypt_inp = nengo.Node(update_next_waypt, size_in=1,
                                size_out=params.sp_dim)
    nengo.Connection(bot_pos_inp, next_waypt_inp, synapse=None)

    # Create state representing next waypoint
    model.next_waypt = spa.State(params.sp_dim, vocab=waypts_vocab)
    nengo.Connection(next_waypt_inp, model.next_waypt.input, synapse=None)

    # Create state representing previous waypoint
    model.prev_waypt = spa.State(params.sp_dim, vocab=waypts_vocab)

    # Create ensemble representing difference between the next and the previous
    # waypoints
    waypts_sps_diff = nengo.Ensemble(400, dimensions=params.sp_dim)
    nengo.Connection(model.next_waypt.output, waypts_sps_diff)
    nengo.Connection(model.prev_waypt.output, waypts_sps_diff, transform=-1.0)
    nengo.Connection(waypts_sps_diff, model.prev_waypt.input)

    # Create cleanup memory for waypoints
    model.waypt_cleanup = spa.AssociativeMemory(
        input_vocab=waypts_vocab, input_keys=waypts_names, wta_output=True)
    nengo.Connection(model.prev_waypt.output, model.waypt_cleanup.input)
    nengo.Connection(model.waypt_cleanup.output, model.prev_waypt.input)

    # Create input initializing previous waypoint
    model.prev_waypt_init = spa.Input(
        prev_waypt=lambda t: waypts_names[0] if t < 0.05 else '0')
