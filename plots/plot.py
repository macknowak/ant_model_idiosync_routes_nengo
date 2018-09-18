import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

params = {
    't_limits': None,
    'plot_bot_pos_aerial.axis': [-8.5, 8.5, -0.5, 8.5],
    'plot_bot_pos_aerial.yticks': np.arange(0.0, 8.1, 1.0)
    }


def plot_action_names(data_path, params):
    """Plot basal ganglia."""
    plt.figure(figsize=(7.8, 4.8))

    t_limits = params['t_limits']

    action_names = np.loadtxt(os.path.join(data_path, "actionname.txt"),
                              dtype=str)
    bg = np.loadtxt(os.path.join(data_path, "bg.txt"))

    plt.plot(bg[:,0], bg[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-0.4, 1.2)
    plt.ylabel("Utility")
    plt.legend(action_names, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("bg")
    plt.tight_layout()


def plot_bot_orient(data_path, params):
    """Plot robot orientations."""
    plt.figure()

    t_limits = params['t_limits']

    bot_orient = np.loadtxt(os.path.join(data_path, "botorient.txt"))
    bot_orient_deg = bot_orient[:,1] * (180.0 / np.pi)

    plt.plot(bot_orient[:,0], bot_orient_deg)
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-180.0, 180.0)
    plt.yticks(np.arange(-180, 181, 45))
    plt.ylabel("Orientation (deg)")
    plt.title("bot_orient")


def plot_bot_pos(data_path, params):
    """Plot robot positions."""
    plt.figure()

    t_limits = params['t_limits']

    bot_pos = np.loadtxt(os.path.join(data_path, "botpos.txt"))

    plt.plot(bot_pos[:,0], bot_pos[:,1], label="x")
    plt.plot(bot_pos[:,0], bot_pos[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("bot_pos")


def plot_bot_pos_aerial(data_path, params):
    """Plot robot positions (aerial view)."""
    plt.figure()

    axis = params['plot_bot_pos_aerial.axis']
    yticks = params['plot_bot_pos_aerial.yticks']

    bot_pos = np.loadtxt(os.path.join(data_path, "botpos.txt"))
    routes_waypts_pos = np.loadtxt(os.path.join(data_path,
                                                "routewayptpos.txt"))
    waypts_pos = np.loadtxt(os.path.join(data_path, "wayptpos.txt"))

    for route_id in range(1, int(routes_waypts_pos[:,0].max()) + 1):
        route_idxs = np.where(routes_waypts_pos[:,0] == route_id)[0]
        plt.plot(routes_waypts_pos[route_idxs,1],
                 routes_waypts_pos[route_idxs,2], ':', linewidth=1.0,
                 color='gray')
    plt.plot(bot_pos[:,1], bot_pos[:,2], '-', linewidth=1.5,
             color='midnightblue')
    plt.plot(waypts_pos[:,0], waypts_pos[:,1], 'o', markersize=5,
             markerfacecolor='green', markeredgecolor='green')
    plt.tick_params(labelsize=8, length=3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis('scaled')
    plt.axis(axis)
    plt.xticks(np.arange(-8.0, 8.1, 1.0))
    plt.yticks(yticks)
    plt.title("bot_pos")


def plot_catch_area(data_path, params):
    """Plot catchment area (waypoint)."""
    plt.figure()

    t_limits = params['t_limits']

    catch_area = np.loadtxt(os.path.join(data_path, "catcharea.txt"))
    waypts_names = np.loadtxt(os.path.join(data_path, "wayptname.txt"),
                              dtype=str)
    n_waypts = len(waypts_names)

    print("Waypoint id (catch_area):")
    for w, waypt_name in enumerate(waypts_names):
        print("{0}: {1}".format(w, waypt_name))
    plt.plot(catch_area[:,0], catch_area[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, n_waypts - 0.5)
    plt.yticks(np.arange(-1, n_waypts))
    plt.ylabel("Waypoint")
    plt.title("catch_area (waypoint_id)")


def plot_catch_area_norm(data_path, params):
    """Plot catchment area (wormalized distance)."""
    plt.figure()

    t_limits = params['t_limits']

    catch_area = np.loadtxt(os.path.join(data_path, "catcharea.txt"))

    plt.plot(catch_area[:,0], catch_area[:,2])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Normalized distance")
    plt.title("catch_area (norm_waypt_dist)")


def plot_catch_vec(data_path, params):
    """Plot catchment vector."""
    plt.figure()

    t_limits = params['t_limits']

    catch_vec = np.loadtxt(os.path.join(data_path, "catchvec.txt"))

    plt.plot(catch_vec[:,0], catch_vec[:,1], label="x")
    plt.plot(catch_vec[:,0], catch_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("catch_vec")


def plot_displace(data_path, params):
    """Plot displacement."""
    plt.figure()

    t_limits = params['t_limits']

    displace = np.loadtxt(os.path.join(data_path, "displace.txt"))

    plt.plot(displace[:,0], displace[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-0.1, 1.1)
    plt.ylabel("State")
    plt.title("displace")


def plot_gain_catch_vec(data_path, params):
    """Plot gain of the catchment vector."""
    plt.figure()

    t_limits = params['t_limits']

    gain_catch_vec = np.loadtxt(os.path.join(data_path, "gaincatchvec.txt"))

    plt.plot(gain_catch_vec[:,0], gain_catch_vec[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Gain")
    plt.title("gain_catch_vec")


def plot_gate_reset(data_path, params):
    """Plot gate for reset signal."""
    plt.figure()

    t_limits = params['t_limits']

    gate_reset = np.loadtxt(os.path.join(data_path, "gatereset.txt"))

    plt.plot(gate_reset[:,0], gate_reset[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("gate_reset")


def plot_gate_routept_cleanup(data_path, params):
    """Plot gate for inhibiting cleanup memory for routepoints."""
    plt.figure()

    t_limits = params['t_limits']

    gate_routept_cleanup = np.loadtxt(os.path.join(data_path,
                                                   "gaterouteptcleanup.txt"))

    plt.plot(gate_routept_cleanup[:,0], gate_routept_cleanup[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("gate_routept_cleanup")


def plot_gate_target_vec(data_path, params):
    """Plot gate for inhibiting target vector."""
    plt.figure()

    t_limits = params['t_limits']

    gate_target_vec = np.loadtxt(os.path.join(data_path, "gatetargetvec.txt"))

    plt.plot(gate_target_vec[:,0], gate_target_vec[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("gate_target_vec")


def plot_gate_target_vec_goal(data_path, params):
    """Plot gate for inhibiting target vector depending on whether current goal
       is set."""
    plt.figure()

    t_limits = params['t_limits']

    gate_target_vec_goal = np.loadtxt(os.path.join(data_path,
                                                   "gatetargetvecgoal.txt"))

    plt.plot(gate_target_vec_goal[:,0], gate_target_vec_goal[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("State")
    plt.title("gate_target_vec_goal")


def plot_gate_view_routept(data_path, params):
    """Plot gate for inhibiting currently viewed routepoint."""
    plt.figure()

    t_limits = params['t_limits']

    gate_view_routept = np.loadtxt(os.path.join(data_path,
                                                "gateviewroutept.txt"))

    plt.plot(gate_view_routept[:,0], gate_view_routept[:,1])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("gate_view_routept")


def plot_global_vec(data_path, params):
    """Plot global vector."""
    plt.figure()

    t_limits = params['t_limits']

    global_vec = np.loadtxt(os.path.join(data_path, "globalvec.txt"))

    plt.plot(global_vec[:,0], global_vec[:,1], label="x")
    plt.plot(global_vec[:,0], global_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("global_vec")


def plot_goal(data_path, params):
    """Plot current goal."""
    plt.figure(figsize=(7.0, 4.8))

    t_limits = params['t_limits']

    goal = np.loadtxt(os.path.join(data_path, "goal.txt"))
    goals_vocab = np.loadtxt(os.path.join(data_path, "goalvocab.txt"),
                             dtype=str)

    plt.plot(goal[:,0], goal[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(goals_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("goal")
    plt.tight_layout()


def plot_goal0view(data_path, params):
    """Plot binder for current goal and current view."""
    plt.figure(figsize=(8.3, 4.8))

    t_limits = params['t_limits']

    goal0view = np.loadtxt(os.path.join(data_path, "goal0view.txt"))
    routepts_vocab = np.loadtxt(os.path.join(data_path, "routeptvocab.txt"),
                                dtype=str)

    plt.plot(goal0view[:,0], goal0view[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(routepts_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("goal0view")
    plt.tight_layout()


def plot_goal_detect(data_path, params):
    """Plot goal detection."""
    plt.figure()

    t_limits = params['t_limits']

    goal_detect = np.loadtxt(os.path.join(data_path, "goaldetect.txt"))

    plt.plot(goal_detect[:,0], goal_detect[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("State")
    plt.title("goal_detect")


def plot_goal_input(data_path, params):
    """Plot new goal."""
    plt.figure(figsize=(7.0, 4.8))

    t_limits = params['t_limits']

    goal_input = np.loadtxt(os.path.join(data_path, "goalinput.txt"))
    goals_vocab = np.loadtxt(os.path.join(data_path, "goalvocab.txt"),
                             dtype=str)

    plt.plot(goal_input[:,0], goal_input[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Similarity")
    plt.legend(goals_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("goal_input")
    plt.tight_layout()


def plot_goals2goals_vecs(data_path, params):
    """Plot associative memory between goals and goal vectors."""
    plt.figure()

    t_limits = params['t_limits']

    goals2goals_vecs = np.loadtxt(os.path.join(data_path, "goal2goalvec.txt"))

    plt.plot(goals2goals_vecs[:,0], goals2goals_vecs[:,1], label="x")
    plt.plot(goals2goals_vecs[:,0], goals2goals_vecs[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("goals2goals_vecs")


def plot_motion_vec(data_path, params):
    """Plot motion vector."""
    plt.figure()

    t_limits = params['t_limits']

    motion_vec = np.loadtxt(os.path.join(data_path, "motionvec.txt"))

    plt.plot(motion_vec[:,0], motion_vec[:,1], label="x")
    plt.plot(motion_vec[:,0], motion_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("motion_vec")


def plot_next_routept(data_path, params):
    """Plot next routepoint."""
    plt.figure(figsize=(8.3, 4.8))

    t_limits = params['t_limits']

    next_routept = np.loadtxt(os.path.join(data_path, "nextroutept.txt"))
    routepts_vocab = np.loadtxt(os.path.join(data_path, "routeptvocab.txt"),
                                dtype=str)

    plt.plot(next_routept[:,0], next_routept[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(routepts_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("next_routept")
    plt.tight_layout()


def plot_norm_target_vec(data_path, params):
    """Plot normalized target vector."""
    plt.figure()

    t_limits = params['t_limits']

    norm_target_vec = np.loadtxt(os.path.join(data_path, "normtargetvec.txt"))

    plt.plot(norm_target_vec[:,0], norm_target_vec[:,1], label="x")
    plt.plot(norm_target_vec[:,0], norm_target_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("norm_target_vec")


def plot_prev2next_routepts(data_path, params):
    """Plot associative memory between previous and next routepoints."""
    plt.figure(figsize=(8.3, 4.8))

    t_limits = params['t_limits']

    prev2next_routepts = np.loadtxt(os.path.join(data_path,
                                                 "prev2nextroutept.txt"))
    routepts_vocab = np.loadtxt(os.path.join(data_path, "routeptvocab.txt"),
                                dtype=str)

    plt.plot(prev2next_routepts[:,0], prev2next_routepts[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(routepts_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("prev2next_routepts")
    plt.tight_layout()


def plot_prev_routept(data_path, params):
    """Plot previous routepoint."""
    plt.figure(figsize=(8.3, 4.8))

    t_limits = params['t_limits']

    prev_routept = np.loadtxt(os.path.join(data_path, "prevroutept.txt"))
    routepts_vocab = np.loadtxt(os.path.join(data_path, "routeptvocab.txt"),
                                dtype=str)

    plt.plot(prev_routept[:,0], prev_routept[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(routepts_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("prev_routept")
    plt.tight_layout()


def plot_routept_cleanup(data_path, params):
    """Plot cleanup memory for routepoints."""
    plt.figure(figsize=(8.3, 4.8))

    t_limits = params['t_limits']

    routept_cleanup = np.loadtxt(os.path.join(data_path, "routeptcleanup.txt"))
    routepts_vocab = np.loadtxt(os.path.join(data_path, "routeptvocab.txt"),
                                dtype=str)

    plt.plot(routept_cleanup[:,0], routept_cleanup[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(routepts_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("routept_cleanup")
    plt.tight_layout()


def plot_routepts2norm_local_vecs(data_path, params):
    """Plot associative memory between routepoints and normalized local vectors
       between route waypoints."""
    plt.figure()

    t_limits = params['t_limits']

    routepts2norm_local_vecs = np.loadtxt(
        os.path.join(data_path, "routept2normlocalvec.txt"))

    plt.plot(routepts2norm_local_vecs[:,0], routepts2norm_local_vecs[:,1],
             label="x")
    plt.plot(routepts2norm_local_vecs[:,0], routepts2norm_local_vecs[:,2],
             label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("routepts2norm_local_vecs")


def plot_routepts2views(data_path, params):
    """Plot associative memory between routepoints and views."""
    plt.figure(figsize=(7.8, 4.8))

    t_limits = params['t_limits']

    routepts2views = np.loadtxt(os.path.join(data_path, "routept2view.txt"))
    views_vocab = np.loadtxt(os.path.join(data_path, "viewvocab.txt"),
                             dtype=str)

    plt.plot(routepts2views[:,0], routepts2views[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(views_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("routepts2views")
    plt.tight_layout()


def plot_scaled_catch_vec(data_path, params):
    """Plot scaled catchment vector."""
    plt.figure()

    t_limits = params['t_limits']

    scaled_catch_vec = np.loadtxt(os.path.join(data_path,
                                               "scaledcatchvec.txt"))

    plt.plot(scaled_catch_vec[:,0], scaled_catch_vec[:,1], label="x")
    plt.plot(scaled_catch_vec[:,0], scaled_catch_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("scaled_catch_vec")


def plot_scaled_local_vec(data_path, params):
    """Plot scaled local vector."""
    plt.figure()

    t_limits = params['t_limits']

    scaled_local_vec = np.loadtxt(os.path.join(data_path,
                                               "scaledlocalvec.txt"))

    plt.plot(scaled_local_vec[:,0], scaled_local_vec[:,1], label="x")
    plt.plot(scaled_local_vec[:,0], scaled_local_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("scaled_local_vec")


def plot_scaled_target_vec(data_path, params):
    """Plot scaled target vector."""
    plt.figure()

    t_limits = params['t_limits']

    scaled_target_vec = np.loadtxt(os.path.join(data_path,
                                                "scaledtargetvec.txt"))

    plt.plot(scaled_target_vec[:,0], scaled_target_vec[:,1], label="x")
    plt.plot(scaled_target_vec[:,0], scaled_target_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("scaled_target_vec")


def plot_target_vec(data_path, params):
    """Plot target vector."""
    plt.figure()

    t_limits = params['t_limits']

    target_vec = np.loadtxt(os.path.join(data_path, "targetvec.txt"))

    plt.plot(target_vec[:,0], target_vec[:,1], label="x")
    plt.plot(target_vec[:,0], target_vec[:,2], label="y")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (m)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("target_vec")


def plot_thal(data_path, params):
    """Plot thalamus."""
    plt.figure(figsize=(7.8, 4.8))

    t_limits = params['t_limits']

    action_names = np.loadtxt(os.path.join(data_path, "actionname.txt"),
                              dtype=str)
    thal = np.loadtxt(os.path.join(data_path, "thal.txt"))

    plt.plot(thal[:,0], thal[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-0.4, 1.2)
    plt.ylabel("Activation")
    plt.legend(action_names, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("thal")
    plt.tight_layout()


def plot_view(data_path, params):
    """Plot current view."""
    plt.figure(figsize=(7.8, 4.8))

    t_limits = params['t_limits']

    view = np.loadtxt(os.path.join(data_path, "view.txt"))
    views_vocab = np.loadtxt(os.path.join(data_path, "viewvocab.txt"),
                             dtype=str)

    plt.plot(view[:,0], view[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Similarity")
    plt.legend(views_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("view")
    plt.tight_layout()


def plot_view7view(data_path, params):
    """Plot comparator between current and retrieved views."""
    plt.figure()

    t_limits = params['t_limits']

    view7view = np.loadtxt(os.path.join(data_path, "view7view.txt"))

    plt.plot(view7view[:,0], view7view[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Similarity")
    plt.title("view7view")


def plot_view_routept(data_path, params):
    """Plot currently viewed routepoint."""
    plt.figure(figsize=(8.3, 4.8))

    t_limits = params['t_limits']

    routepts_vocab = np.loadtxt(os.path.join(data_path, "routeptvocab.txt"),
                                dtype=str)
    view_routept = np.loadtxt(os.path.join(data_path, "viewroutept.txt"))

    plt.plot(view_routept[:,0], view_routept[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.5, 1.5)
    plt.ylabel("Similarity")
    plt.legend(routepts_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("view_routept")
    plt.tight_layout()


def plot_vision(data_path, params):
    """Plot vision."""
    plt.figure(figsize=(7.8, 4.8))

    t_limits = params['t_limits']

    views_vocab = np.loadtxt(os.path.join(data_path, "viewvocab.txt"),
                             dtype=str)
    vision = np.loadtxt(os.path.join(data_path, "vision.txt"))

    plt.plot(vision[:,0], vision[:,1:])
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylim(-1.1, 1.1)
    plt.ylabel("Similarity")
    plt.legend(views_vocab, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("vision")
    plt.tight_layout()


def plot_wheel_speeds(data_path, params):
    """Plot wheel speeds."""
    plt.figure(figsize=(6.8, 4.8))

    t_limits = params['t_limits']

    wheel_speeds = np.loadtxt(os.path.join(data_path, "wheelspeed.txt"))

    plt.plot(wheel_speeds[:,0], wheel_speeds[:,1], label="left")
    plt.plot(wheel_speeds[:,0], wheel_speeds[:,2], label="right")
    if t_limits is not None:
        plt.xlim(t_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.title("wheel_speeds")
    plt.tight_layout()


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
        if 'plot_{}'.format(plotname) not in globals():
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
if not plotnames or 'bot_pos_aerial' in plotnames:
    plot_bot_pos_aerial(data_path, params)
if not plotnames or 'bot_pos' in plotnames:
    plot_bot_pos(data_path, params)
if not plotnames or 'bot_orient' in plotnames:
    plot_bot_orient(data_path, params)
if not plotnames or 'global_vec' in plotnames:
    plot_global_vec(data_path, params)
if not plotnames or 'goal_input' in plotnames:
    plot_goal_input(data_path, params)
if not plotnames or 'goal' in plotnames:
    plot_goal(data_path, params)
if not plotnames or 'gate_target_vec_goal' in plotnames:
    plot_gate_target_vec_goal(data_path, params)
if not plotnames or 'gate_target_vec' in plotnames:
    plot_gate_target_vec(data_path, params)
if not plotnames or 'gate_routept_cleanup' in plotnames:
    plot_gate_routept_cleanup(data_path, params)
if not plotnames or 'catch_area' in plotnames:
    plot_catch_area(data_path, params)
if not plotnames or 'catch_area_norm' in plotnames:
    plot_catch_area_norm(data_path, params)
if not plotnames or 'vision' in plotnames:
    plot_vision(data_path, params)
if not plotnames or 'view' in plotnames:
    plot_view(data_path, params)
if not plotnames or 'goal0view' in plotnames:
    plot_goal0view(data_path, params)
if not plotnames or 'gate_view_routept' in plotnames:
    plot_gate_view_routept(data_path, params)
if not plotnames or 'view_routept' in plotnames:
    plot_view_routept(data_path, params)
if not plotnames or 'prev_routept' in plotnames:
    plot_prev_routept(data_path, params)
if not plotnames or 'routept_cleanup' in plotnames:
    plot_routept_cleanup(data_path, params)
if not plotnames or 'prev2next_routepts' in plotnames:
    plot_prev2next_routepts(data_path, params)
if not plotnames or 'next_routept' in plotnames:
    plot_next_routept(data_path, params)
if not plotnames or 'goal_detect' in plotnames:
    plot_goal_detect(data_path, params)
if not plotnames or 'routepts2views' in plotnames:
    plot_routepts2views(data_path, params)
if not plotnames or 'view7view' in plotnames:
    plot_view7view(data_path, params)
if not plotnames or 'gain_catch_vec' in plotnames:
    plot_gain_catch_vec(data_path, params)
if not plotnames or 'goals2goals_vecs' in plotnames:
    plot_goals2goals_vecs(data_path, params)
if not plotnames or 'target_vec' in plotnames:
    plot_target_vec(data_path, params)
if not plotnames or 'norm_target_vec' in plotnames:
    plot_norm_target_vec(data_path, params)
if not plotnames or 'scaled_target_vec' in plotnames:
    plot_scaled_target_vec(data_path, params)
if not plotnames or 'routepts2norm_local_vecs' in plotnames:
    plot_routepts2norm_local_vecs(data_path, params)
if not plotnames or 'scaled_local_vec' in plotnames:
    plot_scaled_local_vec(data_path, params)
if not plotnames or 'catch_vec' in plotnames:
    plot_catch_vec(data_path, params)
if not plotnames or 'scaled_catch_vec' in plotnames:
    plot_scaled_catch_vec(data_path, params)
if not plotnames or 'motion_vec' in plotnames:
    plot_motion_vec(data_path, params)
if not plotnames or 'wheel_speeds' in plotnames:
    plot_wheel_speeds(data_path, params)
if not plotnames or 'displace' in plotnames:
    plot_displace(data_path, params)
if not plotnames or 'gate_reset' in plotnames:
    plot_gate_reset(data_path, params)
if not plotnames or 'action_names' in plotnames:
    plot_action_names(data_path, params)
if not plotnames or 'thal' in plotnames:
    plot_thal(data_path, params)

# Show figures
plt.show()
