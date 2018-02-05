"""Switching gate states.

This script provides a Nengo model of a gate that can be switched on or off and
maintains its state afterwards. The action selection circuit is needed only to
produce transition to a new state and is not involved in maintaining the new
state later. The gate state can be switched using on and off signals, which
must be manually changed using Nengo GUI.
"""

__author__ = "Przemyslaw (Mack) Nowak"

import nengo
import nengo.presets as presets
import nengo.spa as spa

# Create Nengo model
model = spa.SPA()
with model:
    # Create node representing on signal
    on_inp = nengo.Node(0)

    # Create node representing off signal
    off_inp = nengo.Node(0)

    # Create state representing on signal
    model.on = spa.State(1)
    nengo.Connection(on_inp, model.on.input, synapse=None)

    # Create state representing off signal
    model.off = spa.State(1)
    nengo.Connection(off_inp, model.off.input, synapse=None)

    # Create gate
    with presets.ThresholdingEnsembles(0.3):
        model.gate = spa.State(1, feedback=1)

    # Create action selection circuit
    actions = spa.Actions(
        'on --> gate = 1.0',
        'off --> gate = -1.0',
        '0.5 -->')
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
