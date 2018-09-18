"""Navigation to a sequence of goals."""

# Experiment version
experiment_version = '1.0'

# Location parameters
scene_radius = 10.0
catch_area_radius = 1.0
catch_vec_thres = 0.2
catch_vec_core = 0.2
goals = ["Feed1", "Nest"]
init_global_vec = None
bot_init_pos = None
bot_init_orient = None

# Nengo model parameters
sp_dim = 64
sp_max_similarity = 0.1
action_n2pr_coeff_gvr = 0.2
action_v2pr_coeff_gvr = 0.5
action_v2pr_coeff_intercept = 0.2

# V-REP remote API parameters
vrep_ip = '127.0.0.1'
vrep_port = 19997

# Simulation parameters
nengo_backend = 'nengo'
scene_filename = "scene.ttt"
sim_duration = 65.0  # s
sim_cycle_duration = 0.1  # s
nengo_sim_dt = 0.001  # s

# Random number generator parameters
np_seed = 2062361305

# Other parameters
saved_data = [
    'action_names', 'bg', 'bot_orient', 'bot_pos', 'catch_area', 'catch_vec',
    'displace', 'gain_catch_vec', 'gate_reset', 'gate_routept_cleanup',
    'gate_target_vec', 'gate_target_vec_goal', 'gate_view_routept',
    'global_vec', 'goal', 'goal0view', 'goal_detect', 'goal_input',
    'goals2goals_vecs', 'goals_names', 'goals_vocab', 'motion_vec',
    'next_routept', 'norm_target_vec', 'prev2next_routepts', 'prev_routept',
    'routept_cleanup', 'routepts2norm_local_vecs', 'routepts2views',
    'routepts_vocab', 'routes_waypts_names', 'routes_waypts_pos',
    'scaled_catch_vec', 'scaled_local_vec', 'scaled_target_vec', 'target_vec',
    'thal', 'view', 'view7view', 'view_routept', 'views_vocab', 'vision',
    'waypts_names', 'waypts_pos', 'wheel_speeds'
    ]
prb_syn = 0.005
