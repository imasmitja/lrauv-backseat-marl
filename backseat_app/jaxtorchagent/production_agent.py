import numpy as np
#from .torch_agent import TorchCentralizedActorRNN, load_params
from typing import Optional, List, Tuple

try:
    from backseat_app.jaxtorchagent.tracking.tracker import Tracker_ivan
except:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "backseat_app"))
    from jaxtorchagent.tracking.tracker import Tracker_ivan

try:
    from backseat_app.jaxtorchagent.torch_agent_v3 import CentralizedActorRNN as TorchCentralizedActorRNN
except:
    from jaxtorchagent.torch_agent_v3 import CentralizedActorRNN as TorchCentralizedActorRNN

class TrajectorySmoother:
    def __init__(self, max_speed=20, transition_time=2.0, initial_step_ratio=100, growth_rate=1.2):
        self.max_speed = max_speed
        self.transition_time = transition_time
        self.previous_position = None
        self.in_transition = False
        self.transition_points_x = []
        self.transition_points_y = []
        self.transition_index = 0
        self.smoothed_pos_x = 0
        self.smoothed_pos_y = 0
        self.initial_step_ratio = initial_step_ratio
        self.growth_rate = growth_rate

    def sigmoid_transition(self, start_point, end_point, num_points, steepness=6):
        """
        Sigmoid function for very smooth start and end
        steepness: higher = more abrupt transition in middle
        """
        t = np.linspace(-int(steepness/2), int(steepness/2), int(num_points))
        sigmoid = 1 / (1 + np.exp(-t))
        
        positions = start_point + sigmoid * (end_point - start_point)
        return positions
    
    def generate_progressive_steps(self, start, end, num_points):
        """Generate steps that start small and grow progressively"""
        total_distance = end - start
        
        # Generate geometrically increasing step sizes
        step_sizes = []
        current_step = total_distance * self.initial_step_ratio
        
        for i in range(num_points):
            step_sizes.append(current_step)
            current_step *= self.growth_rate
        
        # Normalize to cover exact distance
        total_steps = sum(step_sizes)
        scaling_factor = total_distance / total_steps
        step_sizes = [s * scaling_factor for s in step_sizes]
        
        # Build positions
        positions = [start]
        current_pos = start
        for step in step_sizes:
            current_pos += step
            positions.append(current_pos)
        
        return positions
    
    def update(self, current_position, new_range = False, dt=0.1):
        if self.previous_position is None:
            self.previous_position = current_position.copy()
            return current_position[0], current_position[1], False
        
        # Calculate instantaneous speed
        speed = abs(np.sqrt((current_position[0] - self.previous_position[0])**2+(current_position[1] - self.previous_position[1])**2)) / dt
        print('speed:',speed)
        if speed > self.max_speed and not self.in_transition:
            #self.previous_position = current_position.copy()
            print('')
            print('--------------------------------------------------------------------')
            print('WARNING SMOOTHING IN ACACTION: prediciton speed = ', speed)
            print('--------------------------------------------------------------------')
            print('')
            #return current_position[0], current_position[1], True
            # Start transition
            self.in_transition = True
            num_points = int(self.transition_time / dt)
            self.transition_points_x = np.linspace(self.previous_position[0],current_position[0], num_points)
            self.transition_points_y = np.linspace(self.previous_position[1],current_position[1], num_points)
            #Method 3: Sigmoid Transition (Smooth Acceleration)
            #self.transition_points_x = self.generate_progressive_steps(self.previous_position[0],current_position[0], num_points)
            #self.transition_points_y = self.generate_progressive_steps(self.previous_position[1],current_position[1], num_points)
            
            self.transition_index = 1
            print('WARNING: SMOOTHING THE PF TARGET ESTIMATION')
            print('transition_points_x:',self.transition_points_x)
            print('transition_points_y:',self.transition_points_y)
            return self.transition_points_x[0],self.transition_points_y[0], False
        
        
        elif self.in_transition:
            # Continue transition
            if self.transition_index < len(self.transition_points_x):
                self.smoothed_pos_x = self.transition_points_x[self.transition_index]
                self.smoothed_pos_y = self.transition_points_y[self.transition_index]
                self.transition_index += 1
                print('WARNING SMOOTHING IN TRANSITION')
                self.previous_position = np.array([self.smoothed_pos_x, self.smoothed_pos_y])
                return self.smoothed_pos_x, self.smoothed_pos_y, False
            else:
                # Transition complete
                self.in_transition = False
                self.transition_points_x = []
                self.transition_points_y = []
                #self.previous_position = np.array([self.smoothed_pos_x, self.smoothed_pos_y])
                #self.previous_position = current_position.copy()
                #return self.smoothed_pos_x, self.smoothed_pos_y, False
                return current_position[0], current_position[1], False
        else:
            # Normal tracking
            self.previous_position = current_position.copy()
            return current_position[0], current_position[1], False

class AgentProductionController:

    def __init__(self, actor, trackers, num_agents=1):
        self.actor = actor
        self.trackers = trackers
        self.smoothers = []
        self.smoother = False
        self.smoothers  = [
        TrajectorySmoother(max_speed=10, transition_time=800) for _ in range(len(self.trackers))
        ]
        #New from DEC 2025 MARL
        self.num_agents = num_agents
        # Store last observed positions of other agents (used when communication is not available)
        self.last_other_agent_obs = {}

    def reset(self, seed=None):
        """Reset the agent and tracker states."""
        self.actor.reset(seed)
        self.smoothers  = [
        TrajectorySmoother(max_speed=10, transition_time=800) for _ in range(len(self.trackers))
        ]
        for tracker in self.trackers:
            tracker.reset()
        #New from DEC 2025 MARL
        self.last_other_agent_obs = {}

    def get_action_and_predictions(
        self,
        ranges: List[List[float]],
        positions: List[Tuple[float, float, float]],
        targets_depth: List[float],
        dt: int = 30,
        new_range: bool = True,
    ):
        """
        angle: yaw of the agent in radians
        ranges (num_landmarks, num_agents): observed distance in meters from landmarks, 0 if not observed,
        positions (num_agents, 3): known agent positions (x, y, z),
        targets_depths (num_landmarks,): known target depths (z),

        For ranges and positions, it is assumed that the first agent is the one using the controller.
        """

        # update tracking for each target
        preds = {}
        testrange=False #to update the PF every time .jaxtorchagewith or without new range, if no range only update, if range update and resample.
        for i, tracker in enumerate(self.trackers):

            #Create a mask to eliminate the ranges and associated position with 0s, as they don't need to be used for trget position estimation
            
            ###test delate
            # mask = np.array([True,True])
            # ranges=np.array([[100,40],[1000,2200]])
            # positions=np.array([[0,0,0],[500,0,0]])

            positions = np.array(positions)

            print('ranges: ', ranges)

            mask = np.array(ranges[i]) != 0
            print('mask before: ', mask)
            mask = mask.flatten()

            #continue
            print('mask: ', mask)
            print('ranges[i]: ', np.array(ranges[i]).flatten())
            print('positions: ', positions)
            # ranges_good = ranges[i][mask.squeeze()]
            # positions_good = positions[mask.squeeze(),:] 
            ranges_good = np.array(ranges[i]).flatten()[mask]
            positions_good = np.array(positions[mask])
            print('ranges_good: ', ranges_good)
            print('positions_good: ', positions_good)
            print('targets_depth: ', targets_depth)

            if len(ranges_good) != 0 or testrange==True:
                #print('INFO: Target prediction True')
                pred = tracker.update_and_predict(
                    agent_pos=positions[0], # only the first agent position is used to initialize the PF
                    ranges=ranges_good,
                    positions=positions_good,
                    depth=targets_depth[i],
                    dt=dt,
                    new_range=new_range
                )
                preds[f'landmark_{i}_tracking_x'] = pred[0]
                preds[f'landmark_{i}_tracking_y'] = pred[1]
                preds[f'landmark_{i}_tracking_z'] = pred[2]
            else:
                if tracker.pred[0] != 0 or tracker.pred[1] != 0:
                    #if no new measurements, I use the old ones
                    #print('INFO: Target prediction False: use old prediction')
                    preds[f'landmark_{i}_tracking_x'] = tracker.pred[0]
                    preds[f'landmark_{i}_tracking_y'] = tracker.pred[1]
                    preds[f'landmark_{i}_tracking_z'] = tracker.pred[2]
                else:
                    #if the prediction is not available, use the first position
                    #print('INFO: Target prediction False: use current position')
                    preds[f'landmark_{i}_tracking_x'] = positions[0][0]
                    preds[f'landmark_{i}_tracking_y'] = positions[0][1]
                    preds[f'landmark_{i}_tracking_z'] = positions[0][2]
                    tracker.pred[0] = positions[0][0]
                    tracker.pred[1] = positions[0][1]
                    tracker.pred[2] = positions[0][2]

            if self.smoother == True:
                #Smoth the predictions to help the MARL agent:
                aux_target = np.array([preds[f'landmark_{i}_tracking_x'] ,preds[f'landmark_{i}_tracking_y'] ])
                smoothed_pos_x, smoothed_pos_y, reset_marl = self.smoothers[i].update(aux_target,new_range=new_range,dt=dt)
                preds[f'landmark_{i}_tracking_x'] = smoothed_pos_x +0.
                preds[f'landmark_{i}_tracking_y'] = smoothed_pos_y +0.
                if reset_marl == True:
                    #self.actor.reset(seed=10, pos=[positions[0][0],positions[0][1],0.])
                    self.actor.reset(seed=10)
        
        
        # prepare the observation for the agent
        obs = {
            "x": positions[0][0],
            "y": positions[0][1],
            "z": 0,  # this can always be 0 for now
            "rph_z": 0,  # this can always be 0
        }

        #New from DEC 2025 MARL
        # logic is that the first agent is the one using the controller
        # Use fresh data if communication is available, otherwise use cached observations
        for j in range(1, self.num_agents):
            if len(positions) > j:
                # Fresh data from communication
                dx = positions[j][0] - positions[0][0]
                dy = positions[j][1] - positions[0][1]
                dz = 0  # this can always be 0 for now since agents are supposed at surface
                # Cache this observation for future use
                self.last_other_agent_obs[f"agent_{j}_dx"] = dx
                self.last_other_agent_obs[f"agent_{j}_dy"] = dy
                self.last_other_agent_obs[f"agent_{j}_dz"] = dz
            else:
                # Use cached data from last communication
                dx = self.last_other_agent_obs.get(f"agent_{j}_dx", 0.0)
                dy = self.last_other_agent_obs.get(f"agent_{j}_dy", 0.0)
                dz = self.last_other_agent_obs.get(f"agent_{j}_dz", 0.0)
            
            obs.update(
                {
                    f"agent_{j}_dx": dx,
                    f"agent_{j}_dy": dy,
                    f"agent_{j}_dz": dz,
                }
            )

        # # OLD stuff
        # # logic is that the first agent is the one using the controller
        # for j in range(1, len(positions)):
        #     if positions[j].sum()!=0:
        #         obs.update(
        #             {
        #                 f"agent_{j}_dx": positions[j][0] - positions[0][0],
        #                 f"agent_{j}_dy": positions[j][1] - positions[0][1],
        #                 f"agent_{j}_dz": 0,  # this can always be 0 for now since agents are supposed at surface
        #             })
        #     else: # if we don't have others agents, we put 0s
        #         obs.update({
        #             f'agent_{j}_dx': 0.,
        #             f'agent_{j}_dy': 0.,
        #             f'agent_{j}_dz': 0.,
        #         })



        for i in range(len(self.trackers)):
            #if ranges[i][0]==0:
               # ranges[i][0] = np.sqrt((positions[0][0]-preds[f'landmark_{i}_tracking_x'])**2+(positions[0][1]-preds[f'landmark_{i}_tracking_y'])**2)
            obs.update(
                {
                    f"landmark_{i}_tracking_x": preds[f"landmark_{i}_tracking_x"],
                    f"landmark_{i}_tracking_y": preds[f"landmark_{i}_tracking_y"],
                    f"landmark_{i}_tracking_z": preds[f"landmark_{i}_tracking_z"],
                    f"landmark_{i}_range": 0,  # this can always be 0 for now
                }
            )

        obs = {"agent_0": obs}  # make the observation compatible with the actor

        print('step:',obs)
        action = self.actor.step(obs, done=False)

        return action["agent_0"], preds


def load(
    num_agents: int,
    num_landmarks: int,
    model_path=None,
    dt=30,
    **tracking_kwargs,
):

    agent = TorchCentralizedActorRNN(
        seed=0,
        agent_params_path=model_path,
        agent_list=[f"agent_{i}" for i in range(num_agents)],
        landmark_list=[f"landmark_{i}" for i in range(num_landmarks)],
        actors_list=["agent_0"], # control only the first agent
        action_dim=5,
        hidden_dim=64,
        pos_norm=1e-3,
        agent_class="ppo_transformer",
        mask_ranges=True,
        matrix_obs=True,
        add_agent_id=False,
        max_range_dist=1500.0,
        normalize_distances=True,
        num_layers=2,
        num_heads=8,
        ff_dim=128,
    )
    
    trackers = [
        Tracker_ivan(method="pf", dt=dt, **tracking_kwargs) for _ in range(num_landmarks)
    ]

    return AgentProductionController(agent, trackers, num_agents=num_agents)


def test(
    model_path: str = "x",
):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import pandas as pd
    import utm
    import os

    episodes = 1
    num_agents = 2
    num_targets = 1
    steps = 200
    #agent_version = "1v1"
    agent_version = "2v1"


    agent_velocity = 1.0  # m/s
    target_velocity = [0.5 * agent_velocity, 0* agent_velocity] # m/s
    step_time = 30  # seconds
    velocity_noise_std = 0.1 * agent_velocity  # m/s
    target_max_depth = 20.0  # meters
    max_initial_distance = 300.0  # meters
    range_error_std = 10.0  # errors in the meters
    new_range_interval = 4  # update the PF only every n steps (my pf doesn't work well with large intervals)
    new_communication_interval = 6  # agents exchange information every n steps
    output_dir = f"outputs/plots_multiagent_agents_test2v2"
    os.makedirs(output_dir, exist_ok=True)

    if agent_version == "1v1":        
        #1- First we load the RL agent
        #This is the agent we tested on Octover 2025 and worked great 1v1
        model_name = "mappo_transformer_1v1_v4.safetensors" # New agent with should work: mappo_transformer_noisy_more_linear_utracking_1_vs_1_step456_rng1948878966
    elif agent_version == "2v1":
        model_name = "mappo_2v1_2december.safetensors" #Same agent as previouse but trained for 2v1
    else:
        print ('ERROR. AGENT VERSION NEED TO BE SPECIFIED CORRECTLY')
        return None
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    project_root = os.path.abspath(project_root) 
    model_path = os.path.join(project_root,"jaxtorchagent", "IROS_MODELS", model_name)

    # Create independent controller for each agent
    agent_controllers = [
        load(
            num_agents=num_agents,
            num_landmarks=num_targets,
            model_path=model_path,
            dt=30,
        )
        for _ in range(num_agents)
    ]

    discrete_action_mapping = np.array([-0.24, -0.12, 0, 0.12, 0.24])
    discrete_action_mapping_heading = np.array([0.903, 0.452, 0.00, -0.452, -0.903])


    print(f"\nTesting with {num_agents} agents and {num_targets} landmarks...")


    first_ranges_mean_all=[]
    first_ranges_std_all=[]
    middle_ranges_mean_all=[]
    middle_ranges_std_all=[]
    last_ranges_mean_all=[]
    last_ranges_std_all=[]
    noise_values = []
    speed_values = []

    #noise value
    noise_value = 0.5 #0.5
    speed_value = 2000
    #for noise_value in np.linspace(0,2,2):
    #for speed_value in [100., 1000., 2000., 3000., 4000., 5000.]:
    for speed_value in [6000.]:

        first_ranges_mean=[]
        first_ranges_std=[]
        middle_ranges_mean=[]
        middle_ranges_std=[]
        last_ranges_mean=[]
        last_ranges_std=[]
        aux_speed = []

        for e in range(episodes):

            np.random.seed(e)
            for controller in agent_controllers:
                controller.reset(seed=e)

            # initial positions for all agents - spawn within max_initial_distance from origin
            agent_positions_init = []
            agent_headings_init = []
            for i in range(num_agents):
                if i == 0:
                    # First agent at origin
                    agent_positions_init.append(np.array([0.0, 0.0, 0.0]))
                else:
                    # Other agents at random positions within max_initial_distance
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(0, max_initial_distance)
                    x = distance * np.cos(angle)
                    y = distance * np.sin(angle)
                    agent_positions_init.append(np.array([x, y, 0.0]))
                agent_headings_init.append(np.random.uniform(0, 2 * np.pi))

            # target position: random position within max_initial_distance sphere at depth
            target_positions_init = []
            target_headings_init = []
            all_target_depth = []
            for i in range(num_targets):
                target_depth = np.random.uniform(5.0, target_max_depth)
                angle_xy = np.random.uniform(0, 2 * np.pi)
                distance_xy = np.random.uniform(100.0, max_initial_distance)
                target_x = distance_xy * np.cos(angle_xy)
                target_y = distance_xy * np.sin(angle_xy)
                if i==1:
                    target_x += 300.
                    target_y += 300.
                #target_pos = np.array([target_x, target_y, target_depth])
                heading = np.random.uniform(0, 2 * np.pi)
                target_positions_init.append(np.array([target_x, target_y, target_depth]))
                target_headings_init.append(heading)
                all_target_depth.append(target_depth)

            # tracking data (now as lists of lists for multiple agents)
            all_agent_positions = [[pos.copy()] for pos in agent_positions_init]
            all_agent_headings = [[heading] for heading in agent_headings_init]
            all_actions_taken = [[] for _ in range(num_agents)]
            all_predictions = [[[] for _ in range(num_targets)] for _ in range(num_agents)]
            all_agent_velocities = [[agent_velocity] for _ in range(num_agents)]
            
            all_target_positions = [[target_pos.copy()] for target_pos in target_positions_init]
            all_target_headings = [[target_heading] for target_heading in agent_headings_init]
            all_target_velocities = [[target_velocity[i]] for i in range(num_targets)]
            
            # Ranges and errors (per agent)
            all_real_ranges = [[[] for _ in range(num_targets)] for _ in range(num_agents)]
            all_measured_ranges = [[[] for _ in range(num_targets)]for _ in range(num_agents)]
            all_tracking_errors = [[[] for _ in range(num_targets)] for _ in range(num_agents)]
            all_distances = [[[] for _ in range(num_targets)] for _ in range(num_agents)]

            print(f"\n=== Episode {e} ===")
            for i in range(num_agents):
                print(f"Initial agent {i} position: {agent_positions_init[i]}")
            for i in range(num_targets):
                print(f"Initial target {i} position: {target_positions_init[i]}")

            for step in range(steps):

                # 1- Check if agents communicate this step
                communication_step = step % new_communication_interval == 0
                update_ranges = step % new_range_interval == 0

                # 2- Get current positions of all agents and targets
                current_agent_positions = [all_agent_positions[i][-1] for i in range(num_agents)]
                current_target_positions = [all_target_positions[i][-1] for i in range(num_targets)]
                
                # 3- Calculate ranges for all agents to target
                current_ranges_all = [[] for _ in range(num_agents)]
                current_measured_ranges_all = [[] for _ in range(num_agents)]
                for i in range(num_agents):
                    for j in range(num_targets):
                        real_range = np.linalg.norm(current_target_positions[j] - current_agent_positions[i])
                        all_real_ranges[i][j].append(real_range)
                        
                        if update_ranges:
                            measured_range = real_range + np.random.normal(0, range_error_std)
                            measured_range = max(0, measured_range)
                            all_measured_ranges[i][j].append(measured_range)
                            current_ranges_all[i].append(measured_range)
                            current_measured_ranges_all[i].append(measured_range)
                        else:
                            all_measured_ranges[i][j].append(0.0)
                            current_ranges_all[i].append(0.0)
                            current_measured_ranges_all[i].append(0.0)

                print('current_ranges_all: ', current_ranges_all)
                # 4- Each agent computes action independently
                actions_this_step = []
                for i in range(num_agents):
                    # first the current agent position and ranges
                    positions = [current_agent_positions[i]]
                    ranges = [[current_ranges_all[i]]]
                    for j in range(num_agents):
                        if j != i:
                            if communication_step:
                                positions.append(current_agent_positions[j])
                                ranges[0].append(current_ranges_all[j])
                            else:
                                positions.append(np.array([0.0, 0.0, 0.0]))  # dummy position
                                ranges[0].append(np.zeros(num_targets))  # dummy range
                    ranges = ranges[0]  # flatten

                    #adjust ranges and positions shapes to have [[range_agent0_target0, range_agent1_target0,...], [range_agent0_target1, ...], ...]
                    ranges = np.matrix(ranges).T
                    positions = np.matrix(positions) 
                    print('ranges: ', ranges )
                    print('positions: ', positions ) 
                    print('all_target_depth: ', all_target_depth )
                    print('step_time: ', step_time )
                    print('update_ranges: ', update_ranges )

                    action, preds = agent_controllers[i].get_action_and_predictions(
                        ranges=ranges,
                        positions=positions,
                        targets_depth=all_target_depth,
                        dt=step_time,
                        new_range=update_ranges,
                    )
                    
                    actions_this_step.append(action)
                    all_actions_taken[i].append(action)
                    
                    for j in range(num_targets):
                        pred_pos = np.array([
                            preds[f"landmark_{j}_tracking_x"],
                            preds[f"landmark_{j}_tracking_y"],
                            preds[f"landmark_{j}_tracking_z"],
                        ])
                        all_predictions[i][j].append(pred_pos)
                    
                        # calculate tracking error (2D error in xy plane)
                        tracking_error = np.linalg.norm(pred_pos[:2] - current_target_positions[j][:2])
                        all_tracking_errors[i][j].append(tracking_error)
                        
                        # calculate distance between agent and target
                        distance = np.linalg.norm(current_agent_positions[i] - current_target_positions[j])
                        all_distances[i][j].append(distance)

                # 5- simulate movement for all agents based on their actions
                for i in range(num_agents):
                    action = actions_this_step[i]
                    agent_heading = all_agent_headings[i][-1]
                    agent_pos = current_agent_positions[i].copy()
                    
                    # action is discrete: 0-4 mapping to heading changes
                    heading_change = discrete_action_mapping_heading[action]
                    agent_heading += heading_change
                    agent_heading = agent_heading % (2 * np.pi)  # normalize to [0, 2π]
                    all_agent_headings[i].append(agent_heading)

                    # add velocity noise
                    current_agent_velocity = agent_velocity + np.random.normal(
                        0, velocity_noise_std
                    )
                    current_agent_velocity = max(
                        0, current_agent_velocity
                    )  # ensure non-negative
                    all_agent_velocities[i].append(current_agent_velocity)

                    # update agent position
                    agent_pos[0] += current_agent_velocity * step_time * np.cos(agent_heading)
                    agent_pos[1] += current_agent_velocity * step_time * np.sin(agent_heading)
                    # z stays at 0 (surface)
                    
                    all_agent_positions[i].append(agent_pos.copy())

                # 6- simulate target movement (linear trajectory with noise)
                for j in range(num_targets):
                    current_target_velocity = target_velocity[j]  # no noise here
                    all_target_velocities[j].append(current_target_velocity)

                    # target heading changes slightly over time (add small random walk)
                    # target_heading += np.random.normal(0, 0.05)
                    # target_heading = target_heading % (2 * np.pi)

                    target_heading = all_target_headings[j][-1]
                    target_pos = current_target_positions[j].copy()

                    # update target position
                    target_pos[0] += (
                        current_target_velocity * step_time * np.cos(target_heading)
                    )
                    target_pos[1] += (
                        current_target_velocity * step_time * np.sin(target_heading)
                    )
                    # z (depth) stays constant

                    # store target trajectory
                    all_target_positions[j].append(target_pos.copy())
                    all_target_headings[j].append(target_heading)


            # convert to numpy arrays for plotting
            all_target_positions_np = [np.array(all_target_positions[j])  for j in range(num_targets)]
            all_target_velocities_np =[np.array(all_target_velocities[j]) for j in range(num_targets)]
            all_agent_positions_np =  [np.array(all_agent_positions[i])   for i in range(num_agents)]
            all_actions_taken_np =    [np.array(all_actions_taken[i])     for i in range(num_agents)] 
            all_agent_velocities_np = [np.array(all_agent_velocities[i])  for i in range(num_agents)]
            all_predictions_np =     [[np.array(all_predictions[i][j])     for j in range(num_targets)] for i in range(num_agents)]
            all_real_ranges_np =     [[np.array(all_real_ranges[i][j])     for j in range(num_targets)] for i in range(num_agents)]
            all_measured_ranges_np = [[np.array(all_measured_ranges[i][j]) for j in range(num_targets)] for i in range(num_agents)]
            all_tracking_errors_np = [[np.array(all_tracking_errors[i][j]) for j in range(num_targets)] for i in range(num_agents)]
            all_distances_np =       [[np.array(all_distances[i][j])       for j in range(num_targets)] for i in range(num_agents)]

            # print summary statistics
            print(f"\n=== Episode {e} Summary ===")
            for i in range(num_agents):
                for j in range(num_targets):
                    print(f"\nAgent {i} - Target {j}:")
                    print(f"  Final agent position: {all_agent_positions_np[i][-1]}")
                    print(f"  Final distance to target: {all_distances_np[i][j][-1]:.2f} m")
                    print(f"  Mean tracking error: {np.mean(all_tracking_errors_np[i][j]):.2f} m")
                    print(f"  Median tracking error: {np.median(all_tracking_errors_np[i][j]):.2f} m")
                    print(f"  Max tracking error: {np.max(all_tracking_errors_np[i][j]):.2f} m")
                    print(f"  Mean distance: {np.mean(all_distances_np[i][j]):.2f} m")
                    print(f"\nFinal target position: {all_target_positions_np[j][-1]}")

            # plot the episode results for each agent
            for i in range(num_agents):
                plot_episode_results(
                    agent_positions=all_agent_positions_np[i],
                    target_positions=all_target_positions_np,
                    predictions=all_predictions_np[i],
                    actions_taken=all_actions_taken_np[i],
                    real_ranges=all_real_ranges_np[i],
                    measured_ranges=all_measured_ranges_np[i],
                    agent_velocities=all_agent_velocities_np[i],
                    target_velocities=all_target_velocities_np,
                    tracking_errors=all_tracking_errors_np[i],
                    distances=all_distances_np[i],
                    step_time=step_time,
                    output_dir=output_dir,
                    episode=e,
                    agent_id=i,
                )
            
            # Plot combined trajectories for all agents
            plot_multiagent_trajectories(
                all_agent_positions=all_agent_positions_np,
                all_target_positions=all_target_positions_np,
                all_predictions=all_predictions_np,
                step_time=step_time,
                output_dir=output_dir,
                episode=e,
                num_agents=num_agents,
                num_targets=num_targets,    
            )

            print(f"Plots saved to {output_dir}/")
            plt.show(block=True)


def plot_episode_results(
    agent_positions,
    target_positions,
    predictions,
    actions_taken,
    real_ranges,
    measured_ranges,
    agent_velocities,
    target_velocities,
    tracking_errors,
    distances,
    step_time,
    output_dir,
    episode,
    agent_id=0,
):
    """Create and save all plots for an episode and agent."""
    import matplotlib.pyplot as plt

    colors = plt.cm.Pastel2(np.linspace(0, 1, len(target_velocities)))

    time_steps = (
        np.arange(len(agent_positions) - 1) * step_time / 60
    )  # convert to minutes
    time_steps_actions = np.arange(len(actions_taken)) * step_time / 60

    # Plot 1: Trajectories in XY plane with predictions
    plt.figure(figsize=(12, 10))
    plt.plot(
        agent_positions[:, 0], agent_positions[:, 1], "b-", label="Agent", linewidth=2
    )

    for i in range(len(target_positions)):
        plt.plot(
            target_positions[i][:, 0],
            target_positions[i][:, 1],
            linestyle="-",
            color=colors[i],
            label=f"Target {i}",
            linewidth=2,
        )
        plt.plot(
            predictions[i][:, 0],
            predictions[i][:, 1],
            linestyle="--",
            color=colors[i],
            label=f"Predictions {i}",
            alpha=0.6,
            linewidth=1,
        )

    # Mark start and end positions
    plt.scatter(
        agent_positions[0, 0],
        agent_positions[0, 1],
        c="blue",
        s=200,
        marker="o",
        edgecolors="black",
        linewidths=2,
        label="Agent Start",
        zorder=5,
    )
    plt.scatter(
        agent_positions[-1, 0],
        agent_positions[-1, 1],
        c="blue",
        s=200,
        marker="X",
        edgecolors="black",
        linewidths=2,
        label="Agent End",
        zorder=5,
    )

    for i in range(len(target_positions)):
        plt.scatter(
            target_positions[i][0, 0],
            target_positions[i][0, 1],
            color=colors[i],
            s=200,
            marker="o",
            edgecolors="black",
            linewidths=2,
            label=f"Target {i} Start",
            zorder=5,
        )
        plt.scatter(
            target_positions[i][-1, 0],
            target_positions[i][-1, 1],
            color=colors[i],
            s=200,
            marker="X",
            edgecolors="black",
            linewidths=2,
            label=f"Target {i} End",
            zorder=5,
        )

    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.title(
        f"Episode {episode} Agent {agent_id}: Trajectories and Predictions (XY Plane)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{output_dir}/episode_{episode}_agent_{agent_id}_trajectories.png", dpi=150)
    plt.close()

    # Plot 2: Agent actions over time
    plt.figure(figsize=(14, 4))
    plt.plot(time_steps_actions, actions_taken, "o-", markersize=3, linewidth=1)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Action (0-4)", fontsize=12)
    plt.title(
        f"Episode {episode} Agent {agent_id}: Actions Over Time", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.yticks([0, 1, 2, 3, 4], ["Left Max", "Left", "Straight", "Right", "Right Max"])
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{output_dir}/episode_{episode}_agent_{agent_id}_actions.png", dpi=150)
    plt.close()

    # Plot 3: Range measurements vs real ranges
    plt.figure(figsize=(14, 5))
    for i in range(len(real_ranges)):
        plt.plot(time_steps, real_ranges[i], linestyle="-", color=colors[i],label=f"Real Range To Target {i}", linewidth=2)
        measured_indices = np.where(measured_ranges[i] > 0)[0]
        if len(measured_indices) > 0:
            plt.scatter(
                time_steps[measured_indices],
                measured_ranges[i][measured_indices],
                color=colors[i],
                s=30,
                label=f"Measured Range to Target {i}",
                alpha=0.7,
                zorder=5,
            )
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Range (m)", fontsize=12)
    plt.title(f"Episode {episode} Agent {agent_id}: Range Measurements", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{output_dir}/episode_{episode}_agent_{agent_id}_ranges.png", dpi=150)
    plt.close()

    # Plot 4: Velocities over time
    plt.figure(figsize=(14, 5))
    plt.plot(
        time_steps, agent_velocities[:-1], "b-", label="Agent Velocity", linewidth=2
    )
    for i in range(len(target_velocities)):
        plt.plot(
            time_steps, target_velocities[i][:-1], linestyle="-", color=colors[i], label=f"Target Velocity {i}", linewidth=2
        )
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Velocity (m/s)", fontsize=12)
    plt.title(
        f"Episode {episode} Agent {agent_id}: Velocities Over Time", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{output_dir}/episode_{episode}_agent_{agent_id}_velocities.png", dpi=150)
    plt.close()

    # Plot 5: Tracking error and distance over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    for i in range(len(tracking_errors)):
        ax1.plot(time_steps, tracking_errors[i], color=colors[i], linestyle="-", linewidth=2)
        ax1.set_xlabel("Time (minutes)", fontsize=12)
        ax1.set_ylabel("Tracking Error (m)", fontsize=12)
        ax1.set_title("Tracking Error Over Time", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(
            y=np.mean(tracking_errors[i]),
            color=colors[i],
            linestyle="--",
            label=f"Mean {i}: {np.mean(tracking_errors[i]):.2f} m",
            linewidth=2,
        )
        ax1.legend(fontsize=10)

        ax2.plot(time_steps, distances[i], color=colors[i], linestyle="-", linewidth=2)
        ax2.set_xlabel("Time (minutes)", fontsize=12)
        ax2.set_ylabel("Distance (m)", fontsize=12)
        ax2.set_title("Agent-Target Distance Over Time", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(
            y=np.mean(distances[i]),
            color=colors[i],
            linestyle="--",
            label=f"Mean {i}: {np.mean(distances[i]):.2f} m",
            linewidth=2,
        )
        ax2.legend(fontsize=10)

    plt.suptitle(
        f"Episode {episode} Agent {agent_id}: Tracking Performance",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{output_dir}/episode_{episode}_agent_{agent_id}_performance.png", dpi=150)
    plt.close()

    # print(f"Generated 5 plots for episode {episode} agent {agent_id}")


def plot_multiagent_trajectories(
    all_agent_positions,
    all_target_positions,
    all_predictions,
    step_time,
    output_dir,
    episode,
    num_agents,
    num_targets,
):
    """Create combined trajectory plot for all agents."""
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    colorst = plt.cm.Pastel2(np.linspace(0, 1, num_targets))
    colorsp = plt.cm.Dark2(np.linspace(0, 1, num_targets*num_agents))
    
    plt.figure(figsize=(14, 12))
    
    # Plot target trajectory
    for i in range(num_targets):
        target_positions = all_target_positions[i]
        color = colorst[i]
        plt.plot(
            target_positions[:, 0],
            target_positions[:, 1],
            '-',
            color=color,
            label=f'Target {i}',
            linewidth=3,
            zorder=10
        )
        plt.scatter(
            target_positions[0, 0],
            target_positions[0, 1],
            color=color,
            s=300,
            marker='o',
            edgecolors='black',
            linewidths=2,
            label=f'Target {i} Start',
            zorder=15
        )
        plt.scatter(
            target_positions[-1, 0],
            target_positions[-1, 1],
            color=color,
            s=300,
            marker='X',
            edgecolors='black',
            linewidths=2,
            label=f'Target {i} End',
            zorder=15
        )
    # plot predictions
    for i in range(num_agents):
        for j in range(num_targets):
            predictions = all_predictions[i][j]
            color = colorsp[j+i]
            plt.plot(
                predictions[:, 0],
                predictions[:, 1],
                '--',
                color=color,
                alpha=0.7,
                linewidth=1,
                label=f'Agent {i} Predictions Target {j}'
            )
    # Plot each agent
    for i in range(num_agents):
        agent_pos = all_agent_positions[i]
        predictions = all_predictions[i]
        color = colors[i]
        
        plt.plot(
            agent_pos[:, 0],
            agent_pos[:, 1],
            '-',
            color=color,
            label=f'Agent {i}',
            linewidth=2
        )
        plt.scatter(
            agent_pos[0, 0],
            agent_pos[0, 1],
            color=color,
            s=200,
            marker='o',
            edgecolors='black',
            linewidths=2,
            zorder=5
        )
        plt.scatter(
            agent_pos[-1, 0],
            agent_pos[-1, 1],
            color=color,
            s=200,
            marker='X',
            edgecolors='black',
            linewidths=2,
            zorder=5
        )
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(
        f'Episode {episode}: Multi-Agent Trajectories ({num_agents} agents {num_targets} targets)',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{output_dir}/episode_{episode}_multiagent_trajectories.png", dpi=150)
    plt.close()
    
    print(f"Generated multi-agent plot for episode {episode}")


if __name__ == "__main__":
    test()
