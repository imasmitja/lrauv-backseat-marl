import numpy as np
from .torch_agent import CentralizedActorRNN, load_params
from .tracking import Tracker, Tracker_ivan
from typing import Optional, List, Tuple

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
        
        # Convert inputs to numpy arrays if they aren't already
        ranges = np.array(ranges)
        positions = np.array(positions)
        targets_depth = np.array(targets_depth)
        
        # use the positions and ranges received by communications and ignore 0s
        # prepare ranges
        mask = ranges[0] != 0
        if mask.any():  # Check if any ranges are non-zero
            ranges_good = ranges[:, mask]
            positions_good = positions[mask, :] 
        else:
            ranges_good = np.array([]).reshape(ranges.shape[0], 0)
            positions_good = np.array([]).reshape(0, positions.shape[1])
        
        # update tracking for each target 
        preds = {}
        for i, tracker in enumerate(self.trackers):
            if ranges_good.shape[1] > 0 and i < ranges_good.shape[0]:  # Check if we have valid ranges for this target
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
                # Use tracker prediction if available, otherwise use current position
                if hasattr(tracker, 'pred') and tracker.pred is not None and (tracker.pred[0] != 0 or tracker.pred[1] != 0):
                    preds[f'landmark_{i}_tracking_x'] = tracker.pred[0]
                    preds[f'landmark_{i}_tracking_y'] = tracker.pred[1]
                    preds[f'landmark_{i}_tracking_z'] = tracker.pred[2]
                else:
                    preds[f'landmark_{i}_tracking_x'] = positions[0][0]
                    preds[f'landmark_{i}_tracking_y'] = positions[0][1]
                    preds[f'landmark_{i}_tracking_z'] = positions[0][2]

        # prepare the observation for the agent - MAKE SURE THIS IS A DICTIONARY
        obs = {
            'x': float(positions[0][0]),
            'y': float(positions[0][1]),
            'z': float(positions[0][2]),
            'rph_z': float(angle),
        }
        
        # logic is that the first agent is the one using the controller
        for j in range(1, len(positions)):
            if positions[j].sum() != 0:
                obs.update({
                    f'agent_{j}_dx': float(positions[j][0] - positions[0][0]),
                    f'agent_{j}_dy': float(positions[j][1] - positions[0][1]),
                    f'agent_{j}_dz': float(positions[j][2] - positions[0][2]),
                })
            else: # if we don't have others agents, we put 0s
                obs.update({
                    f'agent_{j}_dx': 0.0,
                    f'agent_{j}_dy': 0.0,
                    f'agent_{j}_dz': 0.0,
                })

        for i in range(len(self.trackers)):
            # if range is 0 we fake one using the last target prediction and the current lrauv position
            if ranges[i][0] == 0:
                pred_x = preds.get(f'landmark_{i}_tracking_x', positions[0][0])
                pred_y = preds.get(f'landmark_{i}_tracking_y', positions[0][1])
                ranges[i][0] = np.sqrt((positions[0][0] - pred_x)**2 + (positions[0][1] - pred_y)**2)
            
            obs.update({
                f'landmark_{i}_tracking_x': float(preds[f'landmark_{i}_tracking_x']),
                f'landmark_{i}_tracking_y': float(preds[f'landmark_{i}_tracking_y']),
                f'landmark_{i}_tracking_z': float(preds[f'landmark_{i}_tracking_z']),
                f'landmark_{i}_range': float(ranges[i][0]), # range from the first agent
            })

        # Wrap in agent_0 dictionary - THIS IS CRITICAL
        obs_wrapped = {'agent_0': obs}

        print('obs=', obs_wrapped)

        action = self.actor.step(obs_wrapped, done=False)

        return action['agent_0'], preds



def load(
    num_agents:int,
    num_landmarks:int,
    model_path=None,
    dt=30,
    **tracking_kwargs,
):
    params = load_params(model_path)

    # Extract actor parameters from the flattened structure
    actor_params = {}
    for key, value in params.items():
        if key.startswith('actor,'):
            # Remove the 'actor,' prefix and keep the rest
            new_key = key[6:]  # Remove 'actor,'
            actor_params[new_key] = value
    
    print(f"Extracted {len(actor_params)} actor parameters from {len(params)} total parameters")
    
    if not actor_params:
        raise ValueError("No actor parameters found in the model file!")

    agent = CentralizedActorRNN(
        seed=0,
        agent_params=actor_params,  # Pass the extracted actor parameters
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