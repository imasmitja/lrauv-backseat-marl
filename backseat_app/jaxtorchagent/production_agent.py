import numpy as np
from jaxtorchagent.torch_agent import CentralizedActorRNN as TorchCentralizedActorRNN
#from .torch_agent import TorchCentralizedActorRNN, load_params
from .tracking import Tracker_ivan
from typing import Optional, List, Tuple


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
        angle: float,
        ranges: List[List[float]],
        positions: List[Tuple[float, float, float]],
        targets_depth: List[float],
        dt: int = 30,
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
            "x": positions[0][0],
            "y": positions[0][1],
            "z": positions[0][2],
            "rph_z": angle,
        }

        # logic is that the first agent is the one using the controller
        for j in range(1, len(positions)):
            if positions[j].sum()!=0:
                obs.update(
                    {
                        f"agent_{j}_dx": positions[j][0] - positions[0][0],
                        f"agent_{j}_dy": positions[j][1] - positions[0][1],
                        f"agent_{j}_dz": positions[j][2] - positions[0][2],
                    })
            else: # if we don't have others agents, we put 0s
                obs.update({
                    f'agent_{j}_dx': 0.,
                    f'agent_{j}_dy': 0.,
                    f'agent_{j}_dz': 0.,
                })

        for i in range(len(self.trackers)):
            if ranges[i][0]==0:
                ranges[i][0] = np.sqrt((positions[0][0]-preds[f'landmark_{i}_tracking_x'])**2+(positions[0][1]-preds[f'landmark_{i}_tracking_y'])**2)
            obs.update(
                {
                    f"landmark_{i}_tracking_x": preds[f"landmark_{i}_tracking_x"],
                    f"landmark_{i}_tracking_y": preds[f"landmark_{i}_tracking_y"],
                    f"landmark_{i}_tracking_z": preds[f"landmark_{i}_tracking_z"],
                    f"landmark_{i}_range": ranges[i][0],  # range from the first agent
                }
            )

        obs = {"agent_0": obs}  # make the observation compatible with the actor

        print(obs)
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
            action_dim=5,
            hidden_dim=64,
            matrix_obs=True,
            agent_class="ppo_transformer",
            device="cpu",
        )
    
    trackers = [
        Tracker_ivan(method="pf", dt=dt, **tracking_kwargs) for _ in range(num_landmarks)
    ]

    return AgentProductionController(agent, trackers)


def test(
    model_path: str = "x",
):

    episodes = 2
    steps = 10

    #original name = "mappo_rnn_follow_1v1_10min_training_512steps_utracking_1_vs_1_seed0_vmap0_final.safetensors"
    #model_path = "mappo_rnn_1v1.safetensors"
    #original name = "mappo_transformer_follow_from_1v1_landmarkprop25_1024steps_60ksteps_utracking_1_vs_1_seed0_vmap0.safetensors" #Good for 1target and 1agent
    model_path = "mappo_transformer_1v1.safetensors" #Good for 1target and 1agent
    #original name ="mappo_transformer_tracking_from_1024steps_to_larger_team_utracking_3_vs_1_step24412_rng928981903.safetensors" #Good for 1target and multiple agents
    #model_path = "mappo_transformer_3v1.safetensors" #Good for 1target and multiple agents
    #original name = "mappo_transformer_from_5v5follow_256steps_utracking_5_vs_5_step7320_rng202567368.safetensors"
    #model_path = "mappo_transformer_5v5.safetensors"

    for n in range(1, 3):  # number of agents
        for m in range(1, 3):  # number of landmarks

            print(f"\nTesting with {n} agents and {m} landmarks...")

            agent_controller = load(
                num_agents=n, num_landmarks=m, model_path=model_path, dt=30
            )

            for episode in range(episodes):

                agent_controller.reset(seed=episode)
                print(f" Episode {episode}:")

                for step in range(steps):
                    angle = 0.1 * step  # radians
                    ranges = np.array(
                        [[10.0 + i + step for i in range(n)] for j in range(m)]
                    )  # (m, n)
                    positions = np.array(
                        [[5.0 * i + step, 5.0 * i + step, 0.0] for i in range(n)]
                    )  # (n, 3)
                    targets_depth = [10] * m  # (m,)

                    print(
                        f" Step {step}: Ranges: {ranges}, Positions: {positions}, Depths: {targets_depth}"
                    )

                    action, predictions = agent_controller.get_action_and_predictions(
                        angle=angle,
                        ranges=ranges,
                        positions=positions,
                        targets_depth=targets_depth,
                        dt=30,
                    )

                    print(f" Step {step}: Action: {action}")
                    for key, value in predictions.items():
                        print(f"{key}: {value}")


if __name__ == "__main__":
    test()
