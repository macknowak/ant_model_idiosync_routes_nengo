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
representing the previous waypoint. Additionally, the cleanup memory for
waypoints is transiently inhibited to prevent it from interfering during this
process. This transient inhibition is mediated by an inhibitory ensemble, which
becomes activated by an intermediate excitatory ensemble, which in turn becomes
active when a dedicated gate is switched on by the action selection circuit.
Transience is achieved owing to an internal loop that involves two additional
ensembles: an excitatory one, which is activated by the intermediate excitatory
ensemble, followed by an inhibitory one, which in turn inhibits the
intermediate excitatory ensemble. Because of the extended indirect feedback
inhibition of the inhibitory ensemble, the transient period is long enough for
the update process to complete. The gate that triggers the transient inhibition
is switched on by the action selection circuit when it detects that
dissimilarity between the semantic pointer maintained in the working memory and
the semantic pointer representing the next waypoint, which becomes expressed in
the vicinity of the next waypoint, is high. Apart from switching the gate on,
the action selection circuit also activates an excitatory feedback loop from
the state that expresses the semantic pointer representing the previous
waypoint to itself, which is necessary as otherwise the new previous waypoint
could fade away during the update process due to the transient inhibition of
the cleanup memory for waypoints. Following a route is imitated by advancing
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
params.unitary_sps = True
# ------------------

# Create vocabulary
waypts_vocab = spa.Vocabulary(params.sp_dim, unitary=params.unitary_sps)
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
        input_vocab=waypts_vocab, input_keys=waypts_names, inhibitable=True,
        wta_output=True)
    nengo.Connection(model.prev_waypt.output, model.waypt_cleanup.input)
    nengo.Connection(model.waypt_cleanup.output, model.prev_waypt.input)

    # Create gate for inhibiting cleanup memory for waypoints
    model.gate = spa.State(1)

    # Create intermediate excitatory ensemble
    ens_exc = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(model.gate.output, ens_exc)

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
    nengo.Connection(ens_exc, ens_inh)
    nengo.Connection(ens_inh, model.waypt_cleanup.inhibit, transform=6.0)

    # Create comparator between previous and next waypoints
    model.prev7next_waypt = spa.Compare(params.sp_dim, vocab=waypts_vocab)
    nengo.Connection(model.prev_waypt.output, model.prev7next_waypt.inputA)
    nengo.Connection(model.next_waypt.output, model.prev7next_waypt.inputB)

    # Create action selection circuit
    actions = spa.Actions(
        '1.0 - prev7next_waypt --> gate = 1, prev_waypt = prev_waypt',
        '0.4 -->')
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    # Create input initializing previous waypoint
    model.prev_waypt_init = spa.Input(
        prev_waypt=lambda t: waypts_names[0] if t < 0.05 else '0')

    # Create input initializing gate for inhibiting cleanup memory for
    # waypoints
    gate_init = nengo.Node(lambda t: -0.5 if t < 0.05 else 0.0, size_out=1)
    nengo.Connection(gate_init, model.gate.input, synapse=None)
