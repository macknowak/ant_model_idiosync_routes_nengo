"""Transient inhibition using direct inhibitory feedback loop.

This script provides a Nengo model that produces transient inhibition acting on
a tonically activated ensemble. The tonically activated ensemble mimics an
ensemble that is activated by another pathway. It can be transiently inhibited
by an inhibitory ensemble, which becomes active when a trigger signal occurs.
Transience is achieved owing to a direct inhibitory loop from the inhibitory
ensemble to itself.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo

# Create Nengo model
model = nengo.Network()
with model:
    # Create node representing trigger signal
    inhibit_inp = nengo.Node(0)

    # Create inhibitory ensemble
    ens_inhibit = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(inhibit_inp, ens_inhibit, synapse=None)
    nengo.Connection(ens_inhibit, ens_inhibit.neurons, transform=[[-15.0]]*100,
                     synapse=0.2)

    # Create tonically activated ensemble
    ens_bias = nengo.Node(1)
    ens = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(ens_bias, ens, synapse=None)
    nengo.Connection(ens_inhibit, ens.neurons, transform=[[-5.0]]*100)
