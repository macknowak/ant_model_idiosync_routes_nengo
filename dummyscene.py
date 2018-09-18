"""Substitute for V-REP scene.

This module provides a substitute for V-REP scene that is used during
simulations in which the model is not connected to V-REP.
"""

import numpy as np


class Bot(object):
    """Substitute for robot simulated in V-REP."""

    def __init__(self, init_pos, init_orient):
        self.init_pos = init_pos
        self.init_orient = init_orient


class Scene(object):
    """Substitute for V-REP scene."""

    goals_names = ["Nest", "Feed1"]
    goals_pos = [[0.0, 0.0], [6.0, 8.0]] # array
    waypts_names = goals_names + ["Waypoint1", "Waypoint2", "Waypoint3",
                                  "Waypoint4", "Waypoint5"]
    waypts_pos = goals_pos + [[0.0, 2.5], [3.0, 6.5], [5.5, 5.5], [4.5, 3.0],
                              [3.0, 1.0]] # array
    routes_names = ["Route1", "Route2"]
    routes_waypts_names = [
        ["Nest", "Waypoint1", "Waypoint2", "Feed1"],
        ["Feed1", "Waypoint3", "Waypoint4", "Waypoint5", "Nest"]
        ]
    routes_waypts_pos = [
        [[0.0, 0.0], [0.0, 2.5], [3.0, 6.5], [6.0, 8.0]],
        [[6.0, 8.0], [5.5, 5.5], [4.5, 3.0], [3.0, 1.0], [0.0, 0.0]]
        ] # list of arrays

    def __init__(self, params):
        self.bot = Bot(params.bot_init_pos, params.bot_init_orient)
        self.goals_pos = np.array(self.goals_pos, dtype=float)
        self.waypts_pos = np.array(self.waypts_pos, dtype=float)
        self.routes_waypts_pos = \
            [np.array(route_waypts_pos, dtype=float)
             for route_waypts_pos in self.routes_waypts_pos]
