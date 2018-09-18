"""Proxy for V-REP scene.

This module provides a proxy between the model and the V-REP scene, which
introduces an intermediate level of abstraction and is used during simulations
in which the model interacts with V-REP.
"""

import numpy as np
import vrepsim as vrs


class Scene(object):
    """Proxy for V-REP scene."""

    goals_coll_name = "Goals"
    waypts_coll_name = "Waypoints"
    routes_colls_names_root = "Route"
    bot_name = "Pioneer_p3dx"
    bot_motors_names = ["Pioneer_p3dx_leftMotor", "Pioneer_p3dx_rightMotor"]
    bot_dyn_objs_names = [
        "Pioneer_p3dx_leftMotor", "Pioneer_p3dx_leftWheel",
        "Pioneer_p3dx_rightMotor", "Pioneer_p3dx_rightWheel",
        "Pioneer_p3dx_caster_freeJoint1","Pioneer_p3dx_caster_link",
        "Pioneer_p3dx_caster_freeJoint2", "Pioneer_p3dx_caster_wheel"
    ]

    def __init__(self, vrep_sim):
        self.vrep_sim = vrep_sim

        # Create representation of the robot
        self.bot = vrs.PioneerBot(self.vrep_sim, self.bot_name, None,
                                  self.bot_motors_names)
        self.bot.dyn_objs = None  # monkey patch

        # Retrieve names of goals
        goals_coll = vrs.Collection(self.vrep_sim, self.goals_coll_name)
        self.goals_names = goals_coll.get_names()

        # Retrieve positions of goals
        goals_pos = goals_coll.get_positions()
        self.goals_pos = np.array(goals_pos, dtype=float)[:,:2]

        # Retrieve names of waypoints
        waypts_coll = vrs.Collection(self.vrep_sim, self.waypts_coll_name)
        self.waypts_names = waypts_coll.get_names()

        # Retrieve positions of waypoints
        waypts_pos = waypts_coll.get_positions()
        self.waypts_pos = np.array(waypts_pos, dtype=float)[:,:2]

        # Retrieve names of routes
        self.routes_names = []
        routes_colls = []
        r = 1
        while True:
            route_name = self.routes_colls_names_root + str(r)
            try:
                route_coll = vrs.Collection(self.vrep_sim, route_name)
            except vrs.exceptions.ServerError:
                break
            self.routes_names.append(route_name)
            routes_colls.append(route_coll)
            r += 1

        # Retrieve names of route waypoints
        self.routes_waypts_names = [route_coll.get_names()
                                    for route_coll in routes_colls]

        # Retrieve positions of route waypoints
        self.routes_waypts_pos = \
            [np.array(route_coll.get_positions(), dtype=float)[:,:2]
             for route_coll in routes_colls]

    def move_bot(self, new_pos, new_orient=None):
        """Instantly move robot to a new position."""

        # If necessary, create representations of dynamically simulated child
        # objects of the robot
        if self.bot.dyn_objs is None:
            self.bot.dyn_objs = [vrs.SceneObject(self.vrep_sim, name)
                                 for name in self.bot_dyn_objs_names]

        # Set positions of dynamically simulated child objects of the robot to
        # those that they already hold; this is a workaround to force dynamic
        # reset of those objects (that is to remove them from the dynamics
        # engine) as otherwise they could not be moved along with the robot
        # (they will be added again to the dynamics engine once the robot has
        # been moved to the new position)
        for dyn_obj in self.bot.dyn_objs:
            dyn_obj.set_position(dyn_obj.get_position(), allow_in_sim=True)

        # Move robot to the new position
        new_bot_pos = new_pos + [self.bot.get_position()[2]]
        self.bot.set_position(new_bot_pos, allow_in_sim=True)

        # If necessary, rotate robot to the new orientation
        if new_orient is not None:
            new_bot_orient = self.bot.get_orientation()[0:2] + [new_orient]
            self.bot.set_orientation(new_bot_orient, allow_in_sim=True)
