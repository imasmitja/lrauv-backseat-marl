import numpy as np
from .jax_agent import CentralizedActorRNN, load_params
from .tracking import Tracker
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

        # update tracking for each target
        preds = {}
        for i, tracker in enumerate(self.trackers):
            pred = tracker.update_and_predict(
                ranges=ranges[i], positions=positions, depth=targets_depth[i], dt=dt
            )
            preds[f"landmark_{i}_tracking_x"] = pred[0]
            preds[f"landmark_{i}_tracking_y"] = pred[1]
            preds[f"landmark_{i}_tracking_z"] = pred[2]

        # prepare the observation for the agent
        obs = {
            "x": positions[0][0],
            "y": positions[0][1],
            "z": positions[0][2],
            "rph_z": angle,
        }

        # logic is that the first agent is the one using the controller
        for j in range(1, len(positions)):
            obs.update(
                {
                    f"agent_{j}_dx": positions[j][0] - positions[0][0],
                    f"agent_{j}_dy": positions[j][1] - positions[0][1],
                    f"agent_{j}_dz": positions[j][2] - positions[0][2],
                }
            )

        for i in range(len(self.trackers)):
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
    params = load_params(model_path)

    agent = CentralizedActorRNN(
        seed=0,
        agent_params=params["actor"],
        agent_list=[f"agent_{i}" for i in range(num_agents)],
        landmark_list=[f"landmark_{i}" for i in range(num_landmarks)],
        actors_list=["agent_0"],  # make the actor control only the first agent
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

    trackers = [
        Tracker(method="pf", dt=dt, **tracking_kwargs) for _ in range(num_landmarks)
    ]

    return AgentProductionController(agent, trackers)


def test(
    model_path: str = "IROS_MODELS/mappo_transformer_follow_from_1v1_landmarkprop25_1024steps_60ksteps_utracking_1_vs_1_seed0_vmap0.safetensors",
):

    episodes = 2
    steps = 10

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
