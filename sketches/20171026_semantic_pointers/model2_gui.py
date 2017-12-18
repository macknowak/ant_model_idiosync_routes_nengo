"""Adaptation of semantic pointers for route following.

This script provides a Nengo model that uses semantic pointers to represent
information regarding navigation when following routes. A semantic pointer
representing the current destination and a semantic pointer representing the
previous waypoint are combined in a single compositional representation, from
which they can be independently retrieved. The compositional representation is
created in a hierarchical manner, namely the semantic pointer representing the
current destination is bound to the semantic pointer representing the
destination slot and the semantic pointer representing the previous waypoint is
bound to the semantic pointer representing the waypoint slot, and the two
resulting bindings are added together. Following outbound and inbound routes is
imitated by changing the current destination and the previous waypoint over
time.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo.spa as spa

# --- PARAMETERS ---
class Params(object): pass
params = Params()

# Nengo model parameters
params.sp_dim = 64
# ------------------

# Create vocabulary
vocab = spa.Vocabulary(params.sp_dim)
vocab.extend(['DEST', 'NEST', 'FEED', 'WAYPT', 'W1', 'W2', 'W3'])

# Create Nengo model
model = spa.SPA()
with model:
    # Create state representing destination
    model.dest = spa.State(params.sp_dim, vocab=vocab, feedback=1)

    # Create state representing previous waypoint
    model.prev_waypt = spa.State(params.sp_dim, vocab=vocab, feedback=1)

    # Create state representing compositional representation of the destination
    # and the previous waypoint
    model.conv = spa.State(params.sp_dim, vocab=vocab)

    # Create state representing destination retrieved from the compositional
    # representation
    model.dest_estim = spa.State(params.sp_dim, vocab=vocab)

    # Create state representing previous waypoint retrieved from the
    # compositional representation
    model.prev_waypt_estim = spa.State(params.sp_dim, vocab=vocab)

    # Create action selection circuit
    actions = spa.Actions(
        'conv = DEST * dest + WAYPT * prev_waypt',
        'dest_estim = (conv - (WAYPT * prev_waypt)) * ~DEST',
        'prev_waypt_estim = (conv - (DEST * dest)) * ~WAYPT'
        )
    model.cortical = spa.Cortical(actions)

    # Create input imitating following outbound and inbound routes
    def init_dest(t):
        if t < 0.1:
            return 'FEED'
        elif 1.5 <= t < 1.7:
            return 'NEST'
        else:
            return '0'

    def init_prev_waypt(t):
        if t < 0.1:
            return 'W1'
        elif 0.5 <= t < 0.7:
            return 'W2'
        elif 1.0 <= t < 1.2:
            return 'W3'
        elif 1.5 <= t < 1.7:
            return 'W2'
        elif 2.0 <= t < 2.2:
            return 'W1'
        else:
            return '0'

    model.init = spa.Input(dest=init_dest, prev_waypt=init_prev_waypt)
