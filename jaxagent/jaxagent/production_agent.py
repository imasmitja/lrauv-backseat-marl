import numpy as np
from .jax_agent import CentralizedActorRNN, load_params
from .tracking import Tracker, Tracker_ivan
from typing import Optional, List, Tuple

import jax
jax.config.update('jax_platform_name', 'cpu')

METHOD_TRACKER = 'Ivan'
#METHOD_TRACKER = 'Matteo'

class AgentProductionController:

    def __init__(self, actor, trackers):
        self.actor = actor
        self.trackers = trackers


    def reset(self, seed=None):
        """Reset the agent and tracker states."""
        self.actor.reset(seed)
        for tracker in self.trackers:
            tracker.reset()


    def get_action_and_predictions(
            self,
            angle:float,
            ranges:List[List[float]],
            positions:List[Tuple[float, float, float]],
            targets_depth:List[float],
            dt:int=30,
        ):
        """
        angle: yaw of the agent in radians
        ranges (num_landmarks, num_agents): observed distance in meters from landmarks, 0 if not observed,
        positions (num_agents, 3): known agent positions (x, y, z), 
        targets_depths (num_landmarks,): known target depths (z), 

        For ranges and positions, it is assumed that the first agent is the one using the controller.
        """
        
        # use the positions and ranges received by communications and ignore 0s
        # prepare ranges
        #print('###########################################')
        #print('INFO: ranges     =',ranges)
        #print('INFO: positions     =',positions)

        #Create a mask to eliminate the ranges and associated position with 0s, as they don't need to be used for trget position estimation
        mask = ranges[0] != 0
        ranges_good = ranges[:,mask]
        positions_good = positions[mask,:] 
        #Create a mask to eliminate the psitions equal to 0s, as they don't need to be used to update the agent
        #mask = np.any(positions != 0, axis=1)  # Check if any element in row is non-zero
        #positions_obs = positions[mask]

        
        #print('INFO: Good ranges=',ranges_good)
        #print('INFO: Good positions=',positions_good)
        #print('###########################################')
        
        # update tracking for each target 
        preds = {}
        for i, tracker in enumerate(self.trackers):
            if len(ranges_good[0]) != 0:
                #print('INFO: Target prediction True')
                pred = tracker.update_and_predict(
                    ranges=ranges_good[i],
                    positions=positions_good,
                    depth=targets_depth[i],
                    dt=dt
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

        # prepare the observation for the agent
        obs = {
            'x': positions[0][0],
            'y': positions[0][1],
            'z': positions[0][2],
            'rph_z': angle,
        }
        
        # logic is that the first agent is the one using the controller
        for j in range(1, len(positions)):
            if positions[j].sum()!=0:
                obs.update({
                    f'agent_{j}_dx': positions[j][0] - positions[0][0],
                    f'agent_{j}_dy': positions[j][1] - positions[0][1],
                    f'agent_{j}_dz': positions[j][2] - positions[0][2],
                })
            else: # if we don't have others agents, we put 0s
                obs.update({
                    f'agent_{j}_dx': 0.,
                    f'agent_{j}_dy': 0.,
                    f'agent_{j}_dz': 0.,
                })

        for i in range(len(self.trackers)):
             # if range is 0 we fake one using the last target prediction and the current lrauv position
            if ranges[i][0]==0:
                ranges[i][0] = np.sqrt((positions[0][0]-preds[f'landmark_{i}_tracking_x'])**2+(positions[0][1]-preds[f'landmark_{i}_tracking_y'])**2)
            obs.update({
                f'landmark_{i}_tracking_x': preds[f'landmark_{i}_tracking_x'],
                f'landmark_{i}_tracking_y': preds[f'landmark_{i}_tracking_y'],
                f'landmark_{i}_tracking_z': preds[f'landmark_{i}_tracking_z'],
                f'landmark_{i}_range': ranges[i][0], # range from the first agent
            })

        obs = {'agent_0': obs} # make the observation compatible with the actor

        print('obs=',obs)

        #print(obs)
        action = self.actor.step(obs, done=False)

        return action['agent_0'], preds



def load(
    num_agents:int,
    num_landmarks:int,
    model_path=None,
    dt=30,
    **tracking_kwargs,
):
    params = load_params(model_path)

    agent = CentralizedActorRNN(
        seed=0,
        agent_params=params["actor"],
        agent_list=[f"agent_{i}" for i in range(num_agents)], 
        landmark_list=[f"landmark_{i}" for i in range(num_landmarks)],
        actors_list=["agent_0"], # make the actor control only the first agent
        action_dim=5,
        hidden_dim=64,
        pos_norm=1e-3,
        agent_class="ppo_transformer",
        mask_ranges=True,
        matrix_obs=True,
        add_agent_id=False,
        num_layers=2,
        num_heads=8,
        ff_dim=128,
    )

    # make the actor decentralized
    if METHOD_TRACKER != 'Ivan':
        trackers = [Tracker(method='pf', dt=dt, **tracking_kwargs) for _ in range(num_landmarks)]
    else:
        trackers = [Tracker_ivan(method='pf', dt=dt, **tracking_kwargs) for _ in range(num_landmarks)]

    return AgentProductionController(agent, trackers)


#def test(model_path:str="models/FINAL/mappo_transformer_follow_from_1v1_landmarkprop25_1024steps_60ksteps_utracking_1_vs_1_seed0_vmap0.safetensors"):

def test(model_path:str="IROS_MODELS/mappo_transformer_follow_from_1v1_landmarkprop25_1024steps_60ksteps_utracking_1_vs_1_seed0_vmap0.safetensors"):

    episodes=1
    steps=10

    for n in range(1, 3): # number of agents
        for m in range(1, 3): # number of landmarks


            print(f"\nTesting with {n} agents and {m} landmarks...")

            agent_controller = load(
                num_agents=n,
                num_landmarks=m,
                model_path=model_path,
                dt=30
            )

            for episode in range(episodes):

                agent_controller.reset(seed=episode)
                print(f" Episode {episode}:")

                aa = 0
                positions_total = [[0.0,0.0,0.0],[1000.0,0.0,0.0],[1000.0,1000.0,0.0],[0.0,1000.0,0.0],[0.0,0.0,0.0],[1000.0,0.0,0.0],[1000.0,1000.0,0.0],[0.0,1000.0,0.0],[0.0,0.0,0.0],[1000.0,0.0,0.0],[1000.0,1000.0,0.0],[0.0,1000.0,0.0]]

                for step in range(steps):

                    test_ivan = True
                    if test_ivan == True:
                        angle = 0
                        dist = np.sqrt(500**2+500**2)
                        ranges = np.array([[dist]])
                        positions = np.array([positions_total[aa]])
                        targets_depth = [10]
                        aa+=1
                    
                    else:
                        angle = 0.1 * step  # radians
                        ranges = np.array([[10. + i + step for i in range(n)] for j in range(m)])  # (m, n)
                        positions = np.array([[5.0 * i + step, 5.0 * i + step, 0.] for i in range(n)])  # (n, 3)
                        targets_depth = [10]*m  # (m,)

                        #if step == 0:
                           # ranges[0, 0] = int(0)  # first landmark not observed by the first agent

                    print(f" Step {step}: Ranges: {ranges}, Positions: {positions}, Depths: {targets_depth}")

                    action, predictions = agent_controller.get_action_and_predictions(
                        angle=angle,
                        ranges=ranges,
                        positions=positions,
                        targets_depth=targets_depth,
                        dt=30
                    )

                    print(f" Step {step}: Action: {action}")
                    for key, value in predictions.items():
                        print(f"{key}: {value}")


if __name__ == "__main__":
    test()
