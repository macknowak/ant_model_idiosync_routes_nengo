"""Transient inhibition using excitatory-inhibitory feedback loop.

This script provides a Nengo model that produces transient inhibition acting on
a tonically activated ensemble. The tonically activated ensemble mimics an
ensemble that is activated by another pathway. It can be transiently inhibited
by an inhibitory ensemble, which becomes activated by an intermediate
excitatory ensemble, which in turn becomes active when a trigger signal occurs.
Transience is achieved owing to an inhibitory loop from the inhibitory ensemble
back to the intermediate excitatory ensemble. Because of the indirect feedback
inhibition of the inhibitory ensemble, the transient period is longer compared
with the direct feedback inhibition.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo

# Create Nengo model
model = nengo.Network()
with model:
    # Create node representing trigger signal
    inhibit_inp = nengo.Node(0)

    # Create intermediate excitatory ensemble
    ens_exc = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(inhibit_inp, ens_exc, synapse=None)

    # Create inhibitory ensemble
    ens_inhibit = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(ens_exc, ens_inhibit)
    nengo.Connection(ens_inhibit, ens_exc.neurons, transform=[[-20.0]]*100,
                     synapse=0.2)

    # Create tonically activated ensemble
    ens_bias = nengo.Node(1)
    ens = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(ens_bias, ens, synapse=None)
    nengo.Connection(ens_inhibit, ens.neurons, transform=[[-6.0]]*100)
