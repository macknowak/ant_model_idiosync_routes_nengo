import argparse
import os

import matplotlib.pyplot as plt
import nengo_extras.plot_spikes as neps
import numpy as np

plots = {
    'gain_catch_vec_neurons': "gaincatchvecnrn.npz",
    'goal_neurons': "goalnrn.npz",
    'motion_vec_neurons': "motionvecnrn.npz",
    'next_routept_neurons': "prevrouteptnrn.npz",
    'prev_routept_neurons': "prevrouteptnrn.npz",
    'target_vec_neurons': "targetvecnrn.npz",
    'thal_neurons': "thalnrn.npz",
    'view_neurons': "viewnrn.npz",
    'view_routept_neurons': "viewrouteptnrn.npz"
    }

params = {
    'cluster': True,
    'cluster_spikes.n_plotted_neurons': None,
    'cluster_spikes.n_sampled_neurons': None
    }


def cluster_spikes(t, spikes, params):
    """Cluster spike trains."""
    n_sampled_neurons = params['cluster_spikes.n_sampled_neurons']
    if n_sampled_neurons is None:
        n_sampled_neurons = spikes.shape[1]
    n_plotted_neurons = params['cluster_spikes.n_plotted_neurons']
    if n_plotted_neurons is None:
        n_plotted_neurons = n_sampled_neurons

    return neps.merge(
        *neps.cluster(*neps.sample_by_variance(t, spikes,
                                               num=n_sampled_neurons,
                                               filter_width=0.02),
                      filter_width=0.002),
        num=n_plotted_neurons)


def plot_neurons(plotname, data_path, params):
    """Plot spike raster."""
    plt.figure(figsize=(7.5, 6.5))

    cluster = params['cluster']

    with np.load(os.path.join(data_path, plots[plotname])) as data:
        data_neurons = data['data']

    if cluster:
        neps.plot_spikes(*cluster_spikes(data_neurons[:,0], data_neurons[:,1:],
                                         params))
    else:
        neps.plot_spikes(data_neurons[:,0], data_neurons[:,1:])
    plt.xlabel("Time [s]")
    plt.ylabel("Neuron number")
    plt.title(plotname)


# Process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--param", metavar="PARAM",
    dest='params', action='append',
    help="plot parameter")
parser.add_argument(
    "--plot", metavar="PLOT",
    nargs='+',
    dest='plotnames', action='append',
    help="plot name(s)")
parser.add_argument(
    "data_dirname", metavar="DATADIR",
    help="directory with data")
args = parser.parse_args()
data_path = args.data_dirname
if args.plotnames:
    plotnames = \
        [plotname for plotnames in args.plotnames for plotname in plotnames]
    for plotname in plotnames:
        if plotname not in plots:
            raise ValueError("Plot '{}' is not supported.".format(plotname))
else:
    plotnames = []
if args.params:
    for param in args.params:
        paramname, paramval = map(lambda x: x.strip(), param.split("=", 1))
        if paramname in params:
            try:
                params[paramname] = eval(paramval)
            except Exception:
                raise ValueError("Value of parameter '{}' is invalid."
                                 "".format(paramname))
        else:
            raise ValueError("Parameter '{}' is not supported."
                             "".format(paramname))

# Make plots
if not plotnames or 'goal_neurons' in plotnames:
    plot_neurons("goal_neurons", data_path, params)
if not plotnames or 'view_neurons' in plotnames:
    plot_neurons("view_neurons", data_path, params)
if not plotnames or 'view_routept_neurons' in plotnames:
    plot_neurons("view_routept_neurons", data_path, params)
if not plotnames or 'prev_routept_neurons' in plotnames:
    plot_neurons("prev_routept_neurons", data_path, params)
if not plotnames or 'next_routept_neurons' in plotnames:
    plot_neurons("next_routept_neurons", data_path, params)
if not plotnames or 'target_vec_neurons' in plotnames:
    plot_neurons("target_vec_neurons", data_path, params)
if not plotnames or 'gain_catch_vec_neurons' in plotnames:
    plot_neurons("gain_catch_vec_neurons", data_path, params)
if not plotnames or 'motion_vec_neurons' in plotnames:
    plot_neurons("motion_vec_neurons", data_path, params)
if not plotnames or 'thal_neurons' in plotnames:
    plot_neurons("thal_neurons", data_path, params)

# Show figures
plt.show()
