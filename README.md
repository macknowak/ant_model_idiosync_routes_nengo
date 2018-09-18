# Ant navigation along established idiosyncratic routes, controlled by Nengo

## Model

This Nengo model controls a differential-drive robot simulated using V-REP
simulator, which mimics a desert ant navigating in a foraging area with several
established routes between different goals. Communication with V-REP is handled
using the V-REP remote API.

The robot is supposed to navigate to a particular goal among several ones that
exist in the scene. The scene must also contain routes linking those goals. The
routes can possibly be multi-segment, with successive segments connected at
waypoints. Navigation is based on three kinds of vectors: (1) the global vector
(which points from the current robot position to the origin), (2) local vectors
(each related to a waypoint on a route and pointing to the next waypoint on
that route), and (3) catchment vectors (which point from the current robot
position to the waypoint they are related to). The robot can either rely on
global vector navigation (e.g., when it is in an unfamiliar area) or follow a
route using local vectors. Catchment vectors are employed only when the robot
enters the catchment area of a waypoint and are activated during both global
vector navigation and route following. They imitate guidance based on comparing
the current view with the stored view related to that waypoint as vision is
implemented only in an abstract way. The contributions of the three vectors are
dynamically weighted. In the beginning of simulation, a current goal should be
set (it can be changed at any time). Initially, the robot relies on global
vector navigation, but can switch to route following whenever it encounters a
waypoint belonging to a route that leads to the current goal (which may be
already at the start). As the robot moves, arrivals at waypoints are detected
as is arrival at the goal. Additionally, reaction to a passive displacement can
occur, which involves switching to global vector navigation.

Although the robot model used in V-REP may contain sensors, they are not used
because the robot position is obtained directly and vision is implemented in an
abstract way, being imitated based on that position by representing views as
high-dimensional vectors of arbitrary numbers. The robot model must comprise
2 motor-driven wheels (1 on the left and 1 on the right), each of which
receives a signal controlling its velocity from the Nengo model. The
recommended robot model offered by V-REP is *Pioneer_p3dx*.

This model was developed for Nengo 2.7.0 and V-REP 3.4.0.

## Experiments

The Nengo model is employed in the following experiments.

### Navigation to a single goal

In this experiment the Nengo model controls a differential-drive robot
simulated using V-REP simulator such that it navigates towards a specified goal
in the scene.

The robot operates in a scene that must contain several goals linked by routes.
The routes can possibly be multi-segment, with successive segments connected at
waypoints. The robot model used in V-REP is *Pioneer_p3dx*. It has several
sensors, 2 motor-driven wheels (1 on the left and 1 on the right), and 1 caster
wheel (in the back). None of the sensors are used, however, whereas each
motor-driven wheel receives signals controlling its velocity from the Nengo
model. The initial position and orientation of the robot can optionally be set
prior to simulation. In the beginning of simulation, a current goal is set and
then the robot sets off trying to reach that goal.

To successfully run this experiment, V-REP simulator must first be launched
with a continuous remote API server service started and then scene file
`scene.ttt` must be opened in the simulator. To allow the experiment script to
remotely control the *Pioneer_p3dx* robot that is part of this scene, the
original child script associated with this robot was removed from the scene
file.

Moreover, to successfully run this experiment, the following V-REP files (or,
alternatively, links to them) have to exist in the current directory so that
the V-REP remote API could be used (here, `VREP_DIR` denotes the directory in
which V-REP is installed):

- `vrep.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `vrepConst.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `[remoteApi.dll | remoteApi.dylib | remoteApi.so]` (original file in:
  `VREP_DIR/programming/remoteApiBindings/lib/lib/[32Bit | 64Bit]/`).

This experiment was developed for Nengo 2.7.0 and V-REP 3.4.0.

### Navigation to a sequence of goals

In this experiment the Nengo model controls a differential-drive robot
simulated using V-REP simulator such that it navigates towards several goals in
the scene, specified in a sequence, trying to reach them one by one.

The robot operates in a scene that must contain several goals linked by routes.
The routes can possibly be multi-segment, with successive segments connected at
waypoints. The robot model used in V-REP is *Pioneer_p3dx*. It has several
sensors, 2 motor-driven wheels (1 on the left and 1 on the right), and 1 caster
wheel (in the back). None of the sensors are used, however, whereas each
motor-driven wheel receives signals controlling its velocity from the Nengo
model. The initial position and orientation of the robot can optionally be set
prior to simulation. In the beginning of simulation, the first goal from the
sequence is set as current and then the robot sets off trying to reach that
goal. Once it reaches the goal, the next goal from the sequence is set as
current and the robot again tries to reach it. The whole procedure repeats
until the ultimate goal is reached.

To successfully run this experiment, V-REP simulator must first be launched
with a continuous remote API server service started and then scene file
`scene.ttt` must be opened in the simulator. To allow the experiment script to
remotely control the *Pioneer_p3dx* robot that is part of this scene, the
original child script associated with this robot was removed from the scene
file.

Moreover, to successfully run this experiment, the following V-REP files (or,
alternatively, links to them) have to exist in the current directory so that
the V-REP remote API could be used (here, `VREP_DIR` denotes the directory in
which V-REP is installed):

- `vrep.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `vrepConst.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `[remoteApi.dll | remoteApi.dylib | remoteApi.so]` (original file in:
  `VREP_DIR/programming/remoteApiBindings/lib/lib/[32Bit | 64Bit]/`).

This experiment was developed for Nengo 2.7.0 and V-REP 3.4.0.

### Navigation to a goal interrupted by displacements

In this experiment the Nengo model controls a differential-drive robot
simulated using V-REP simulator such that it navigates towards a specified goal
in the scene, but upon arriving at a particular location, the robot is
displaced to a different release point.

The robot operates in a scene that must contain several goals linked by routes.
The routes can possibly be multi-segment, with successive segments connected at
waypoints. The robot model used in V-REP is *Pioneer_p3dx*. It has several
sensors, 2 motor-driven wheels (1 on the left and 1 on the right), and 1 caster
wheel (in the back). None of the sensors are used, however, whereas each
motor-driven wheel receives signals controlling its velocity from the Nengo
model. The initial position and orientation of the robot can optionally be set
prior to simulation. In the beginning of simulation, a current goal is set and
then the robot sets off trying to reach that goal. However, there is one
specific location in the scene such that if the robot happens to arrive within
a specified radius of that location, it is immediately displaced to a release
point. During a simulation, multiple displacements may occur from that location
as release points are specified in a sequence and employed one by one on
subsequent arrivals (there may be many different release points or many
instances of the same release point; the maximum number of displacements
depends on the length of that sequence). Release points, apart from location,
may optionally define the subsequent robot orientation.

To successfully run this experiment, V-REP simulator must first be launched
with a continuous remote API server service started and then scene file
`scene.ttt` must be opened in the simulator. To allow the experiment script to
remotely control the *Pioneer_p3dx* robot that is part of this scene, the
original child script associated with this robot was removed from the scene
file.

Moreover, to successfully run this experiment, the following V-REP files (or,
alternatively, links to them) have to exist in the current directory so that
the V-REP remote API could be used (here, `VREP_DIR` denotes the directory in
which V-REP is installed):

- `vrep.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `vrepConst.py` (original file in:
  `VREP_DIR/programming/remoteApiBindings/python/python/`);
- `[remoteApi.dll | remoteApi.dylib | remoteApi.so]` (original file in:
  `VREP_DIR/programming/remoteApiBindings/lib/lib/[32Bit | 64Bit]/`).

This experiment was developed for Nengo 2.7.0 and V-REP 3.4.0.
