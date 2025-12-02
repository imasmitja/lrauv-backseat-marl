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

    def __init__(self, actor, trackers):
        self.actor = actor
        self.trackers = trackers
        self.smoothers = []
        self.smoother = False
        self.smoothers  = [
        TrajectorySmoother(max_speed=10, transition_time=800) for _ in range(len(self.trackers))
        ]

    def reset(self, seed=None):
        """Reset the agent and tracker states."""
        self.actor.reset(seed)
        self.smoothers  = [
        TrajectorySmoother(max_speed=10, transition_time=800) for _ in range(len(self.trackers))
        ]
        for tracker in self.trackers:
            tracker.reset()

    def get_action_and_predictions(
        self,
        angle: float,
        ranges: List[List[float]],
        positions: List[Tuple[float, float, float]],
        targets_depth: List[float],
        dt: int = 30,
        new_range: bool = True,
        test_marl: bool = False,
        previous_target: List=[]
    ):
        """
        angle: yaw of the agent in radians
        ranges (num_landmarks, num_agents): observed distance in meters from landmarks, 0 if not observed,
        positions (num_agents, 3): known agent positions (x, y, z),
        targets_depths (num_landmarks,): known target depths (z),

        For ranges and positions, it is assumed that the first agent is the one using the controller.
        """

        #Create a mask to eliminate the ranges and associated position with 0s, as they don't need to be used for trget position estimation
        mask = ranges[0] != 0
        ranges_good = ranges[:,mask]
        positions_good = positions[mask,:] 

        # update tracking for each target
        preds = {}
        testrange=False #to update the PF every time .jaxtorchagewith or without new range, if no range only update, if range update and resample.
        for i, tracker in enumerate(self.trackers):
            if test_marl == True:
                #I use the previouse loaded predictions
                preds[f'landmark_{0}_tracking_x'] = previous_target[0]
                preds[f'landmark_{0}_tracking_y'] = previous_target[1]
                preds[f'landmark_{0}_tracking_z'] = previous_target[2] 
            
            elif len(ranges_good[0]) != 0 or testrange==True:
                #print('INFO: Target prediction True')
                pred = tracker.update_and_predict(
                    agent_pos=positions[0],
                    ranges=ranges_good[i],
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
                    preds[f'landmark_{0}_tracking_x'] = tracker.pred[0]
                    preds[f'landmark_{0}_tracking_y'] = tracker.pred[1]
                    preds[f'landmark_{0}_tracking_z'] = tracker.pred[2]
                else:
                    #if the prediction is not available, use the first position
                    #print('INFO: Target prediction False: use current position')
                    preds[f'landmark_{0}_tracking_x'] = positions[0][0]
                    preds[f'landmark_{0}_tracking_y'] = positions[0][1]
                    preds[f'landmark_{0}_tracking_z'] = positions[0][2]
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

        # logic is that the first agent is the one using the controller
        for j in range(1, len(positions)):
            if positions[j].sum()!=0:
                obs.update(
                    {
                        f"agent_{j}_dx": positions[j][0] - positions[0][0],
                        f"agent_{j}_dy": positions[j][1] - positions[0][1],
                        f"agent_{j}_dz": 0,  # this can always be 0 for now since agents are supposed at surface
                    })
            else: # if we don't have others agents, we put 0s
                obs.update({
                    f'agent_{j}_dx': 0.,
                    f'agent_{j}_dy': 0.,
                    f'agent_{j}_dz': 0.,
                })

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
    agent_version=4,
    **tracking_kwargs,
):
    
    if agent_version == 1: #First agent trained and tested during the first field tests
        try:
            from backseat_app.jaxtorchagent.torch_agent import CentralizedActorRNN as TorchCentralizedActorRNN
        except:
            from jaxtorchagent.torch_agent import CentralizedActorRNN as TorchCentralizedActorRNN
        print ('INFO: USING AGENT VERSION NUMBER 1')
    elif agent_version == 2 or agent_version == 3 : #Second agent trained and tested during the second field tests
        try:
            print('load try agent v2')
            from backseat_app.jaxtorchagent.torch_agent_v2 import CentralizedActorRNN as TorchCentralizedActorRNN
            print('loaded correctly')
        except:
            print('load except agent v2')
            from jaxtorchagent.torch_agent_v2 import CentralizedActorRNN as TorchCentralizedActorRNN
        print ('INFO: USING AGENT VERSION NUMBER 2. This agent we modified the observation vector, now the pos of the agent is always relative to its own previous state')
    else:
        try:
            print('load try agent v3')
            from backseat_app.jaxtorchagent.torch_agent_v3 import CentralizedActorRNN as TorchCentralizedActorRNN
            print('loaded correctly')
        except:
            print('load except agent v3')
            from jaxtorchagent.torch_agent_v3 import CentralizedActorRNN as TorchCentralizedActorRNN
        print ('INFO: USING AGENT VERSION NUMBER 23 This agent we modified the observation vector, now the pos of the agent is always relative to its own previous state')
    


    agent = TorchCentralizedActorRNN(
        seed=0,
        agent_params_path=model_path,
        agent_list=[f"agent_{i}" for i in range(num_agents)],
        landmark_list=[f"landmark_{i}" for i in range(num_landmarks)],
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

    return AgentProductionController(agent, trackers)


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
    num_agents = 1
    num_targets = 1
    use_values_pretested = False
    use_previouse_pf_estimations = True #if false, we will run again the PF to estiamte the positions.
    use_current_action = True #if true, we use the current actions to move the agent
    use_syntetic_data = True #if true, the target position will be generated syntetically
    add_noise = True
    plot_graphs = True
    agent_version = 4

    if agent_version == 1:        
        #1- First we load the RL agent
        #original name = "mappo_rnn_follow_1v1_10min_training_512steps_utracking_1_vs_1_seed0_vmap0_final.safetensors"
        #model_name = "mappo_rnn_1v1.safetensors"
        #original name = "mappo_transformer_follow_from_1v1_landmarkprop25_1024steps_60ksteps_utracking_1_vs_1_seed0_vmap0.safetensors" #Good for 1target and 1agent
        model_name = "mappo_transformer_1v1.safetensors" #Good for 1target and 1agent
        #original name ="mappo_transformer_tracking_from_1024steps_to_larger_team_utracking_3_vs_1_step24412_rng928981903.safetensors" #Good for 1target and multiple agents
        #model_name = "mappo_transformer_3v1.safetensors" #Good for 1target and multiple agents
        #original name = "mappo_transformer_from_5v5follow_256steps_utracking_5_vs_5_step7320_rng202567368.safetensors"
        #model_name = "mappo_transformer_5v5.safetensors"
    elif agent_version == 2:
        model_name = "mappo_transformer_1v1_v2.safetensors" #New agent trained with new observation vector state
    elif agent_version == 3:
        model_name = "mappo_transformer_1v1_v3.safetensors" #New agent trained with new observation vector state
    elif agent_version == 4:
        model_name = "mappo_transformer_1v1_v4.safetensors" # New agent with should work: mappo_transformer_noisy_more_linear_utracking_1_vs_1_step456_rng1948878966
    else:
        print ('ERROR. AGENT VERSION NEED TO BE SPECIFIED CORRECTLY')
        return None
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    project_root = os.path.abspath(project_root) 
    model_path = os.path.join(project_root,"jaxtorchagent", "IROS_MODELS", model_name)


    for n in range(1, num_agents+1):  # number of agents
        for m in range(1, num_targets+1):  # number of landmarks

            print(f"\nTesting with {n} agents and {m} landmarks...")

            agent_controller = load(
                num_agents=n, num_landmarks=m, model_path=model_path, agent_version=agent_version, dt=30
            )


            
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

                for episode in range(episodes):

                    agent_controller.reset(seed=int(np.random.rand()*994))
                    print(f" Episode {episode}:")

                    if use_values_pretested == True:
                        file = "/home/maiv/lrauv_sim/lrauv-backseat-marl/logs/20251110T054802_log.txt" # a good example
                        file = "/home/maiv/lrauv_sim/lrauv-backseat-marl/logs/20251110T185757_log.txt" # a bad trajectory
                        file = "/home/maiv/lrauv_sim/lrauv-backseat-marl/logs/20251110T185813_log.txt" # a bad trajectory

                        
                        data = np.loadtxt(file, delimiter=',')
                        # Extract columns (0-based indexing)
                        range_to_target = data[:, 4]   # Column 5
                        x_positions = data[:, 10]       # Column 10
                        y_positions = data[:, 11]      # Column 11
                        lon_target = data[:, 5]      # Column 11
                        lat_target = data[:, 6]      # Column 11
                        action_used = data[:, 7]
                        x_origin = data[:, 18][0] 
                        y_origin = data[:, 20][0]  

                        x_target = []
                        y_target = []
                        for n in range(len(lon_target)):
                            tuple = utm.from_latlon(lat_target[n], lon_target[n])
                            lrauv_x, lrauv_y, zonenumber, zoneletter = tuple
                            x_target.append(lrauv_x-x_origin)
                            y_target.append(lrauv_y-y_origin)
                        x_target = np.array(x_target)
                        y_target = np.array(y_target)

                    if use_syntetic_data == True:
                        #creat syntetic data
                        aux = 1000 #(np.random.rand()*2-1)*speed_value
                        x_target = np.linspace(0, aux, 800) #* 0 +1
                        aux = 10000 #(np.random.rand()*2-1)*speed_value
                        y_target = np.linspace(0, aux, 800) #* -0. +1.1  #with (*0+1) the policy is not quite good
                        x_target = x_target + 500 #(np.random.rand()*2-1)*500 
                        y_target = y_target + 500 #(np.random.rand()*2-1)*500
                        
                        #add noise
                        y_target += np.random.rand(len(y_target))*5
                        x_target += np.random.rand(len(x_target))*5

                        #x_target[200:400] -= 2000.

                        #y_target[200:] += 2000.

                        aux = y_target[200:].copy()
                        x_target[200:] = aux-2000
                        y_target[200:] = y_target[200:] * 0 + y_target[200]

                        y_target[250:] = aux[50:]
                        x_target[250:] = x_target[250:]*0
                        

                        
                        x_positions = np.zeros(len(x_target))
                        y_positions = np.zeros(len(x_target))
                        range_to_target = np.zeros(len(x_target))
                        action_used = np.zeros(len(x_target))


                    #In Matteo's method a 0 means no measurement.
                    range_to_target = np.where(range_to_target == -1, 0, range_to_target)

                    

                    print('ranges=',range_to_target[:10])
                    print('x_positions=',x_positions[:10])
                    print('y_positions=',y_positions[:10])
                    print('x_target=',x_target[:10])
                    print('y_target=',y_target[:10])
                    steps = len(x_positions)

                    discrete_action_mapping = np.array([-0.24, -0.12, 0, 0.12, 0.24])
                    discrete_action_mapping_heading = np.array([0.903, 0.452, 0.00, -0.452, -0.903])

                    x_pos = []
                    y_pos = []
                    x_pred = []
                    y_pred = []
                    actions = []
                    timestamp = []
                    computed_ranges =[]
                    vehicle_vels = []
                    target_vels = []
                    vehicle_rel_vel = []
                    old_heading_inc = []
                    current_range_compute = []
                    smoothed_positions_x = []
                    smoothed_positions_y = []
                    smoothed_positions_step = []
                    smooth_target_vel = []
                    range_to_pred=[]
                    
                    new_heading=0
                    aux_count = 0
                    for step in range(steps):
                        
                        if use_current_action == True:
                            if len(y_pos)>=1:
                                try:
                                    angle = np.arctan2(y_pos[-1]-y_pos[-2],x_pos[-1]-x_pos[-2])
                                    print('inc_y:',y_pos[-1]-y_pos[-2])
                                    print('inc_x:',x_pos[-1]-x_pos[-2])
                                    #print('angle arctan=',angle*180./np.pi)
                                    #Convert yaw to 0-360 degrees, East reference
                                    if angle <= 0.:
                                        angle = np.pi+(np.pi+angle)
                                    #print('angle 360   =',angle*180./np.pi)
                                    #Add 90 degrees offset to rotate yaw to North reference
                                    angle = angle-(np.pi/2.)
                                    angle = angle% (2.*np.pi)
                                    #print('angle 360  N=',angle*180./np.pi)
                                    # Convert yaw back to +-180 degrees, with North reference
                                    if angle > np.pi:
                                        angle = angle - 2.*np.pi
                                    #we change the sign to make clockwise positive angles. In the first implementation, we saw that the first Gazebo implementation
                                    #uses a counterclockwise convention, so we don't need to negate it here.
                                    angle = -angle +0.
                                    #print('angle+-180 N=',angle*180./np.pi)
                                    #Now back to 360 N but with clockwise positive angles
                                    angle = angle%(2.*np.pi)
                                    #print('Agen yaw (360NorthClockwise)=',angle*180./np.pi)

                                    #compute velocity 
                                    deltatime = 30. #s
                                    vehicle_vel = np.sqrt((x_pos[-2]-x_pos[-1])**2+(y_pos[-2]-y_pos[-1])**2)/deltatime
                                    target_vel = np.sqrt((x_target[step-1]-x_target[step])**2+(y_target[step-1]-y_target[step])**2)/deltatime
                                    real_vel = np.sqrt((x_positions[step]-x_positions[step-1])**2+(y_positions[step]-y_positions[step-1])**2)/deltatime
                                except:
                                    deltatime = 30.
                                    angle = 0
                                    vehicle_vel = 0
                                    target_vel =0

                                #compute new position
                                using_old_actions = False
                                if using_old_actions == True:
                                    aux_action = int(np.where(discrete_action_mapping == np.round(action_used[step-1],2))[0]) #previouse actions used in Gazebo
                                    velocity = real_vel+0.
                                    heading_inc = discrete_action_mapping_heading[aux_action]
                                    heading_inc = -heading_inc+0. #I have to reverse the rudder action, as during training the vehicle angles were counterclockwise.
                                    dynamic_factor = np.mean(old_heading_inc[-2:]) * 0.7
                                else:
                                    aux_action = int(np.where(discrete_action_mapping == np.round(actions[step-1],2))[0]) #current actions computed here
                                    velocity = 1. #m/s
                                    if add_noise == True:
                                        velocity += (np.random.rand()*2-1)*noise_value
                                        velocity = np.clip(velocity,0.2,1.2)
                                    heading_inc = discrete_action_mapping_heading[aux_action]
                                    heading_inc = -heading_inc+0. #I have to reverse the rudder action, as during training the vehicle angles were counterclockwise.
                                    dynamic_factor = 0.
                                    
                                new_heading = (angle + heading_inc + dynamic_factor)%(2*np.pi)
                                print('action_used:',aux_action)
                                print('angle:',angle*180./np.pi)
                                print('heading_inc:',heading_inc*180./np.pi)
                                print('dynamic_factor:',dynamic_factor*180./np.pi)
                                print('new_heading:',new_heading*180./np.pi)
                                positions = np.array([[x_pos[-1]+np.sin(new_heading)*velocity*deltatime, y_pos[-1]+np.cos(new_heading)*velocity*deltatime, 0.0]])
                                #add noise to the position of the vehicle
                                if add_noise == True:
                                    positions[0][:2] += (np.random.rand(2)*2-1)*noise_value*2 #noise_value #1.01, 5.01 are great examples
                                
                            else:
                                angle = 0.1 * step  # radians
                                heading_inc = 0.
                                dynamic_factor = 0.
                                vehicle_vel = 0
                                target_vel = 0
                                positions = np.array([[x_positions[step], y_positions[step], 0.0]])
                                real_vel = 0.

                            current_range_compute.append(np.sqrt((positions[0][0]-x_target[step])**2+(positions[0][1]-y_target[step])**2))
                            
                            ranges = np.array([[current_range_compute[-1]]])
                            vehicle_vels.append(vehicle_vel)
                            target_vels.append(target_vel)
                            vehicle_rel_vel.append(real_vel)
                            old_heading_inc.append(heading_inc)
                            
                            

                        else:
                            angle = 0.1 * step  # radians
                            ranges = np.array([[range_to_target[step]]])
                            positions = np.array([[x_positions[step], y_positions[step], 0.0]])

                        targets_depth = [10] * m  # (m,)

                        #save for ploting
                        x_pos.append(positions[0][0])
                        y_pos.append(positions[0][1])

                        computed_ranges.append(np.sqrt((x_positions[step]-x_target[step])**2+(y_positions[step]-y_target[step])**2))
                        

                        #set if we have new_range, aka different to 0
                        if ranges == 0 or step%10 != 0:
                            new_range = False
                            aux_target = [x_target[aux_count],y_target[aux_count],targets_depth[0]]
                        else:
                            new_range = True
                            aux_target = [x_target[step],y_target[step],targets_depth[0]]
                            aux_count = step+0
                        

                        print(
                            f" Step {step}: Ranges: {ranges}, Positions: {positions}, Depths: {targets_depth}, new_range: {new_range}, aux_target: {aux_target}"
                        )
                        
                        action, predictions = agent_controller.get_action_and_predictions(
                            angle=angle,
                            ranges=ranges,
                            positions=positions,
                            targets_depth=targets_depth,
                            dt=30,
                            new_range=new_range,
                            test_marl=use_previouse_pf_estimations,
                            previous_target=aux_target
                        )

                        actions.append(discrete_action_mapping[action])
                        timestamp.append(step)
                        x_pred.append(predictions["landmark_0_tracking_x"])
                        y_pred.append(predictions["landmark_0_tracking_y"])

                        print(f" Step {step}: Action: {action}")
                        print('')

                        range_to_pred.append(np.sqrt((positions[0][0]-x_pred[-1])**2+(positions[0][1]-y_pred[-1])**2))

                        try:
                            smooth_target_vel.append(float(np.sqrt((x_pred[-1]-x_pred[-2])**2+((y_pred[-1]-y_pred[-2])**2))/deltatime))
                        except:
                            smooth_target_vel.append(0)


                    #compute ranges
                    n = len(current_range_compute)
                    # Calculate split points
                    split1 = n // 3
                    split2 = 2 * n // 3
                    # Split the array
                    first = current_range_compute[:split1]
                    middle = current_range_compute[split1:split2]
                    last = current_range_compute[split2:]
                    #save data for ploting afterwards
                    first_ranges_mean.append(np.mean(first))
                    first_ranges_std.append(np.std(first))
                    middle_ranges_mean.append(np.mean(middle))
                    middle_ranges_std.append(np.std(middle))
                    last_ranges_mean.append(np.mean(last))
                    last_ranges_std.append(np.std(last))
                    aux_speed.append(np.mean(target_vels))

                    if plot_graphs == True:
                        # Create a single figure with 4 subplots
                        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

                        # Subplot 1: Vehicle Movement on X-Y Plane
                        ax1 = axes[0, 0]
                        ax1.plot(x_target, y_target, 'r-', label='Past Target Prediction', linewidth=2)
                        ax1.plot(x_pred, y_pred, 'm^--', label='Current Target Prediction', linewidth=1.5, markersize=4)
                        ax1.plot(smoothed_positions_x, smoothed_positions_y, 'mo--', label='Smoothed Target Prediction', linewidth=1.5, markersize=4)
                        #ax1.plot(x_positions, y_positions, 'mo-', alpha=0.3, label='Past Vehicle Path', linewidth=1)
                        ax1.plot(x_pos, y_pos, 'bo--', label='Current Vehicle Path', linewidth=1.5, markersize=3)
                        ax1.plot(x_pos[-1], y_pos[-1], 'c^', markersize=10, label='Current Position')
                        ax1.set_xlabel('X Position')
                        ax1.set_ylabel('Y Position')
                        ax1.set_title('Vehicle Movement on X-Y Plane. Nose value:%.2f'%noise_value)
                        ax1.grid(True, alpha=0.3)
                        ax1.axis('equal')
                        ax1.legend()

                        # Subplot 2: MARL Actions
                        ax2 = axes[0, 1]
                        # ax2.plot(timestamp, action_used, 'g-', label='Past actions')  # Uncomment if you have this data
                        ax2.plot(timestamp, actions, 'ro-', label='Current Actions', linewidth=2, markersize=4)
                        ax2.set_xlabel('Timestamp')
                        ax2.set_ylabel('Actions')
                        ax2.set_title('MARL Actions Over Time')
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()

                        # Subplot 3: Ranges
                        ax3 = axes[1, 0]
                        # ax3.plot(timestamp, computed_ranges, 'ro--', label='Past Ranges (lrauv-pf) pos')  # Uncomment if you have this data
                        ax3.plot(timestamp, current_range_compute, 'bo-', label='Ranges to Target', linewidth=2, markersize=4)
                        ax3.plot(timestamp, range_to_pred, 'm-', label='Ranges to Smoothed Target')  # Uncomment if you have this data
                        ax3.set_xlabel('Timestamp')
                        ax3.set_ylabel('Range (m)')
                        ax3.set_title('Computed Ranges Over Time')
                        ax3.grid(True, alpha=0.3)
                        ax3.legend()

                        # Subplot 4: Velocities
                        ax4 = axes[1, 1]
                        ax4.plot(timestamp, vehicle_vels, 'g-', label='Current Vehicle Velocity', linewidth=2)
                        ax4.plot(timestamp, target_vels, 'b-', label='Current Target Velocity', linewidth=2)
                        ax4.plot(timestamp, smooth_target_vel, 'm^-', label='Smoothed Target Velocity', linewidth=2)
                        
                        ax4.set_xlabel('Timestamp')
                        ax4.set_ylabel('Velocity (m/s)')
                        ax4.set_title('Vehicle and Target Velocities')
                        ax4.grid(True, alpha=0.3)
                        ax4.legend()

                        # Adjust layout and display
                        plt.tight_layout(pad=4.0)
                        plt.show(block=False)

                #save data for ploting afterwards
                first_ranges_mean_all.append(np.mean(first_ranges_mean))
                first_ranges_std_all.append(np.mean(first_ranges_std))
                middle_ranges_mean_all.append(np.mean(middle_ranges_mean))
                middle_ranges_std_all.append(np.mean(middle_ranges_std))
                last_ranges_mean_all.append(np.mean(last_ranges_mean))
                last_ranges_std_all.append(np.mean(last_ranges_std))
                noise_values.append(noise_value)
                speed_values.append(np.mean(aux_speed))
            
            plot2_graphs = True
            if plot2_graphs == True:
                # Create 4 subplots
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
                fig.suptitle('Range Performance Analysis vs Noise Level', fontsize=16, fontweight='bold')

                # Subplot 1: First Third with shaded area
                ax1.plot(noise_values, first_ranges_mean_all, 'ro-', linewidth=3, markersize=8, label='First Third Mean')
                ax1.fill_between(noise_values, 
                                np.array(first_ranges_mean_all) - np.array(first_ranges_std_all), 
                                np.array(first_ranges_mean_all) + np.array(first_ranges_std_all), 
                                alpha=0.3, color='red', label='First Third ±1 STD')
                ax1.set_xlabel('Noise Level')
                ax1.set_ylabel('Mean Range (m)')
                ax1.set_title('First Third - Shaded STD')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Subplot 2: Middle Third with shaded area
                ax2.plot(noise_values, middle_ranges_mean_all, 'go-', linewidth=3, markersize=8, label='Middle Third Mean')
                ax2.fill_between(noise_values, 
                                np.array(middle_ranges_mean_all) - np.array(middle_ranges_std_all), 
                                np.array(middle_ranges_mean_all) + np.array(middle_ranges_std_all), 
                                alpha=0.3, color='green', label='Middle Third ±1 STD')
                ax2.set_xlabel('Noise Level')
                ax2.set_ylabel('Mean Range (m)')
                ax2.set_title('Middle Third - Shaded STD')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Subplot 3: Last Third with shaded area
                ax3.plot(noise_values, last_ranges_mean_all, 'bo-', linewidth=3, markersize=8, label='Last Third Mean')
                ax3.fill_between(noise_values, 
                                np.array(last_ranges_mean_all) - np.array(last_ranges_std_all), 
                                np.array(last_ranges_mean_all) + np.array(last_ranges_std_all), 
                                alpha=0.3, color='blue', label='Last Third ±1 STD')
                ax3.set_xlabel('Noise Level')
                ax3.set_ylabel('Mean Range (m)')
                ax3.set_title('Last Third - Shaded STD')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # Subplot 4: All thirds with error bars
                # Subplot 4: With shaded areas and colorbar (alternative styling)
                x_positions = [1, 2, 3]
                third_labels = ['First', 'Middle', 'Last']

                # Use plasma colormap for better visual distinction
                colors = cm.plasma(np.linspace(0, 1, len(noise_values)))

                # Plot each noise level
                #for i, noise in enumerate(noise_values):
                for i, noise in enumerate(speed_values):
                    means_for_this_noise = [
                        first_ranges_mean_all[i],
                        middle_ranges_mean_all[i], 
                        last_ranges_mean_all[i]
                    ]
                    
                    stds_for_this_noise = [
                        first_ranges_std_all[i],
                        middle_ranges_std_all[i],
                        last_ranges_std_all[i]
                    ]
                    
                    # Plot mean line with thicker line for better visibility
                    ax4.plot(x_positions, means_for_this_noise, 's-', 
                            linewidth=3, markersize=8, color=colors[i], 
                            markerfacecolor=colors[i], markeredgecolor='white', markeredgewidth=1.5,
                            alpha=0.9)
                    
                    # Add shaded area for ±1 STD with slightly higher alpha
                    ax4.fill_between(x_positions, 
                                    np.array(means_for_this_noise) - np.array(stds_for_this_noise),
                                    np.array(means_for_this_noise) + np.array(stds_for_this_noise),
                                    alpha=0.3, color=colors[i], linewidth=0)

                ax4.set_xlabel('Trajectory Third', fontsize=12, labelpad=10)
                ax4.set_ylabel('Mean Range (m)', fontsize=12, labelpad=10)
                ax4.set_title('Range Performance Across Trajectory Thirds\nby Target Speed Level', 
                            fontsize=13, fontweight='bold', pad=15)
                ax4.set_xticks(x_positions)
                ax4.set_xticklabels(third_labels, fontsize=11)
                ax4.grid(True, alpha=0.3)

                # Add colorbar with better styling
                sm = plt.cm.ScalarMappable(cmap=cm.plasma, 
                                        norm=plt.Normalize(min(speed_values), max(speed_values)))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax4, shrink=0.8, pad=0.02)
                cbar.set_label('Speed Level', fontsize=12, labelpad=10)
                cbar.ax.tick_params(labelsize=10)

                # Add a note about the shading
                ax4.text(0.02, 0.98, 'Shaded: ±1 STD', transform=ax4.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                

                plt.tight_layout(pad=4.0)
                plt.subplots_adjust(top=0.93)
                plt.show(block=False)
            #print('ranges=',range_to_target[:10])
            #print('x_positions=',x_positions[:10])
            #print('y_positions=',y_positions[:10])
            #print('x_target=',x_target[:10])
            #print('y_target=',y_target[:10])
            print('speed_values:',speed_values)
            plt.show(block=True)


if __name__ == "__main__":
    test()
