import numpy as np

from typing import Optional, List, Tuple
import os

try:
    from backseat_app.jaxtorchagent.tracking.tracker import Tracker_ivan as Tracker
    from backseat_app.jaxtorchagent.torch_agent_v3 import CentralizedActorRNN as TorchCentralizedActorRNN
except:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "backseat_app"))
    from jaxtorchagent.tracking.tracker import Tracker_ivan as Tracker
    from jaxtorchagent.torch_agent_v3 import CentralizedActorRNN as TorchCentralizedActorRNN


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_controller")


class AgentProductionController:

    def __init__(self, actor, trackers, fake_reset_period=128, threshold_for_reset=800):
        self.actor = actor
        self.trackers = trackers
        self.fake_reset_period = fake_reset_period
        self.threshold_for_reset = threshold_for_reset

    def reset(self, seed=None):
        """Reset the agent and tracker states."""
        logger.info("Resetting agent and trackers")
        self.actor.reset(seed)
        for tracker in self.trackers:
            tracker.reset()
        self.step_count = 0

    def get_action_and_predictions(
        self,
        ranges: List[List[float]],
        positions: List[Tuple[float, float, float]],
        targets_depth: List[float],
        dt: int = 30,
        update=True,
        test_marl: bool = False,
        previous_target: List=[]
    ):
        """
        ranges (num_landmarks, num_agents): observed distance in meters from landmarks, 0 if not observed,
        positions (num_agents, 3): known agent positions (x, y, z),
        targets_depths (num_landmarks,): known target depths (z), DEPTH IS ASSUMED TO BE KNOWN AND POSITIVE

        For ranges and positions, it is assumed that the first agent is the one using the controller.
        """

        self.step_count += 1
        if self.step_count % self.fake_reset_period == 0:
            self.reset()

        # update tracking for each target
        preds = {}
        for i, tracker in enumerate(self.trackers):
            if test_marl == True:
                #I use the previouse loaded predictions
                pred = [previous_target[0],previous_target[1],previous_target[2]]
            elif update:
                pred = tracker.update_and_predict(
                    agent_pos=positions[0], ranges=ranges[i], positions=positions, depth=targets_depth[i], dt=dt, new_range=update
                )
            else:
                pred = tracker.pred

            preds[f"landmark_{i}_tracking_x"] = pred[0]
            preds[f"landmark_{i}_tracking_y"] = pred[1]
            preds[f"landmark_{i}_tracking_z"] = pred[2]  # targets_depth[i]

        # reset if the predicted position is too far from the agent
        if ranges[0][0] != 0:
            if ranges[0][0] > self.threshold_for_reset:
                self.reset()

        # prepare the observation for the agent
        obs = {
            "x": positions[0][0],
            "y": positions[0][1],
            "z": 0,  # this can always be 0 for now
            "rph_z": 0,  # this can always be 0
        }

        # logic is that the first agent is the one using the controller
        for j in range(1, len(positions)):
            obs.update(
                {
                    f"agent_{j}_dx": positions[j][0] - positions[0][0],
                    f"agent_{j}_dy": positions[j][1] - positions[0][1],
                    f"agent_{j}_dz": 0,  # this can always be 0 for now since agents are supposed at surface
                }
            )

        for i in range(len(self.trackers)):
            obs.update(
                {
                    f"landmark_{i}_tracking_x": preds[f"landmark_{i}_tracking_x"],
                    f"landmark_{i}_tracking_y": preds[f"landmark_{i}_tracking_y"],
                    f"landmark_{i}_tracking_z": preds[f"landmark_{i}_tracking_z"],
                    f"landmark_{i}_range": 0,  # this can always be 0 for now
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
        Tracker(method="pf", dt=dt, **tracking_kwargs) for _ in range(num_landmarks)
    ]

    return AgentProductionController(agent, trackers)


def test(
    model_path: str = "x",
):

    episodes = 1
    steps = 800
    num_agents = 1
    num_landmarks = 1

    agent_velocity = 1.0  # m/s
    target_velocity = 0.6 * agent_velocity  # m/s
    step_time = 30  # seconds
    velocity_noise_std = 0.2 * agent_velocity  # m/s
    target_max_depth = 20.0  # meters
    max_initial_distance = 800.0  # meters
    range_error_std = 10.0  # errors in the meters
    new_range_interval = 1  # update the PF only every n steps (my pf doesn't work well with large intervals)
    use_previouse_pf_estimations = True # #if false, we will run again the PF to estiamte the positions.

    model_name = "mappo_transformer_1v1_v4.safetensors" # New agent with should work: mappo_transformer_noisy_more_linear_utracking_1_vs_1_step456_rng1948878966
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    project_root = os.path.abspath(project_root) 
    model_path = os.path.join(project_root,"jaxtorchagent", "IROS_MODELS", model_name)

    output_dir = "outputs/plots_ivan_simplified"
    os.makedirs(output_dir, exist_ok=True)

    agent_controller = load(
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        model_path=model_path,
        dt=step_time,
    )

    # change in heading for discrete actions (5 discrete actions)
    discrete_action_mapping_heading = np.array([0.903, 0.452, 0.00, -0.452, -0.903])

    for e in range(episodes):

        np.random.seed(e)
        agent_controller.reset(seed=e)

        # initial positions
        agent_pos = np.array([0.0, 0.0, 0.0])  # start at surface (x, y, z)

        # target position: random position within max_initial_distance sphere at depth
        target_depth = np.random.uniform(5.0, target_max_depth)

        # random angle and distance in xy plane
        angle_xy = np.random.uniform(0, 2 * np.pi)
        distance_xy = np.random.uniform(100.0, max_initial_distance)
        target_x = distance_xy * np.cos(angle_xy)
        target_y = distance_xy * np.sin(angle_xy)
        target_pos = np.array([target_x, target_y, target_depth])

        # random initial headings
        agent_heading = np.random.uniform(0, 2 * np.pi)
        target_heading = np.random.uniform(0, 2 * np.pi)

        # tracking data
        agent_positions = [agent_pos.copy()]
        target_positions = [target_pos.copy()]
        agent_headings = [agent_heading]
        target_headings = [target_heading]
        actions_taken = []
        predictions = []
        real_ranges = []
        measured_ranges = []
        agent_velocities = [agent_velocity]
        target_velocities = [target_velocity]
        tracking_errors = []
        distances = []

        print(f"\n=== Episode {e} ===")
        print(f"Initial agent position: {agent_pos}")
        print(f"Initial target position: {target_pos}")
        print(f"Initial distance: {np.linalg.norm(target_pos - agent_pos):.2f} m")

        for step in range(steps):

            # simulate ranges every new_range_interval steps with noise
            update_ranges = step % new_range_interval == 0

            if update_ranges:
                # calculate real range
                real_range = np.linalg.norm(target_pos - agent_pos)
                real_ranges.append(real_range)

                # add noise to range measurement
                measured_range = real_range + np.random.normal(0, range_error_std)
                measured_range = max(0, measured_range)  # ensure non-negative
                measured_ranges.append(measured_range)

                # prepare ranges list for controller (one range per landmark)
                ranges = [[measured_range]]  # [[range for agent 0 to landmark 0]]
            else:
                ranges = [[0.0]]  # no new measurement
                real_ranges.append(np.linalg.norm(target_pos - agent_pos))
                measured_ranges.append(0.0)  # no measurement this step

            # get action and predictions from controller
            positions = [tuple(agent_pos)]  # list of agent positions
            targets_depth = [target_depth]  # list of target depths

            action, preds = agent_controller.get_action_and_predictions(
                ranges=ranges,
                positions=positions,
                targets_depth=targets_depth,
                dt=step_time,
                update=update_ranges,
                test_marl=use_previouse_pf_estimations,
                previous_target=[target_pos[0],target_pos[1],target_depth]
            )

            actions_taken.append(action)
            pred_pos = np.array(
                [
                    preds["landmark_0_tracking_x"],
                    preds["landmark_0_tracking_y"],
                    preds["landmark_0_tracking_z"],
                ]
            )
            predictions.append(pred_pos)

            # calculate tracking error (2D error in xy plane)
            tracking_error = np.linalg.norm(pred_pos[:2] - target_pos[:2])
            tracking_errors.append(tracking_error)

            # calculate distance between agent and target
            distance = np.linalg.norm(agent_pos - target_pos)
            distances.append(distance)

            # simulate agent movement based on action
            # action is discrete: 0-4 mapping to heading changes
            heading_change = discrete_action_mapping_heading[action]
            agent_heading += heading_change
            agent_heading = agent_heading % (2 * np.pi)  # normalize to [0, 2π]

            # add velocity noise
            current_agent_velocity = agent_velocity + np.random.normal(
                0, velocity_noise_std
            )
            current_agent_velocity = max(
                0, current_agent_velocity
            )  # ensure non-negative
            agent_velocities.append(current_agent_velocity)

            # update agent position
            agent_pos[0] += current_agent_velocity * step_time * np.cos(agent_heading)
            agent_pos[1] += current_agent_velocity * step_time * np.sin(agent_heading)
            # z stays at 0 (surface)

            # simulate target movement (linear trajectory with noise)
            current_target_velocity = target_velocity  # no noise here
            target_velocities.append(current_target_velocity)

            # target heading changes slightly over time (add small random walk)
            # target_heading += np.random.normal(0, 0.05)
            # target_heading = target_heading % (2 * np.pi)

            # update target position
            target_pos[0] += (
                current_target_velocity * step_time * np.cos(target_heading)
            )
            target_pos[1] += (
                current_target_velocity * step_time * np.sin(target_heading)
            )
            # z (depth) stays constant
            if step == 200:
                target_pos[0] += 2000
                target_pos[1] += 2000

            # store trajectories
            agent_positions.append(agent_pos.copy())
            target_positions.append(target_pos.copy())
            agent_headings.append(agent_heading)
            target_headings.append(target_heading)

        # convert to numpy arrays for plotting
        agent_positions = np.array(agent_positions)
        target_positions = np.array(target_positions)
        predictions = np.array(predictions)
        actions_taken = np.array(actions_taken)
        real_ranges = np.array(real_ranges)
        measured_ranges = np.array(measured_ranges)
        agent_velocities = np.array(agent_velocities)
        target_velocities = np.array(target_velocities)
        tracking_errors = np.array(tracking_errors)
        distances = np.array(distances)

        # print summary statistics
        print(f"\n=== Episode {e} Summary ===")
        print(f"Final agent position: {agent_positions[-1]}")
        print(f"Final target position: {target_positions[-1]}")
        print(f"Final distance: {distances[-1]:.2f} m")
        print(f"Mean tracking error: {np.mean(tracking_errors):.2f} m")
        print(f"Median tracking error: {np.median(tracking_errors):.2f} m")
        print(f"Max tracking error: {np.max(tracking_errors):.2f} m")
        print(f"Mean distance: {np.mean(distances):.2f} m")

        # plot the episode results
        plot_episode_results(
            agent_positions=agent_positions,
            target_positions=target_positions,
            predictions=predictions,
            actions_taken=actions_taken,
            real_ranges=real_ranges,
            measured_ranges=measured_ranges,
            agent_velocities=agent_velocities,
            target_velocities=target_velocities,
            tracking_errors=tracking_errors,
            distances=distances,
            step_time=step_time,
            output_dir=output_dir,
            episode=e,
        )

        print(f"Plots saved to {output_dir}/")


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
):
    """Create and save all plots for an episode."""

    time_steps = (
        np.arange(len(agent_positions) - 1) * step_time / 60
    )  # convert to minutes
    time_steps_actions = np.arange(len(actions_taken)) * step_time / 60

    # Plot 1: Trajectories in XY plane with predictions
    plt.figure(figsize=(12, 10))
    plt.plot(
        agent_positions[:, 0], agent_positions[:, 1], "b-", label="Agent", linewidth=2
    )
    plt.plot(
        target_positions[:, 0],
        target_positions[:, 1],
        "r-",
        label="Target",
        linewidth=2,
    )
    plt.plot(
        predictions[:, 0],
        predictions[:, 1],
        "g--",
        label="Predictions",
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
    plt.scatter(
        target_positions[0, 0],
        target_positions[0, 1],
        c="red",
        s=200,
        marker="o",
        edgecolors="black",
        linewidths=2,
        label="Target Start",
        zorder=5,
    )
    plt.scatter(
        target_positions[-1, 0],
        target_positions[-1, 1],
        c="red",
        s=200,
        marker="X",
        edgecolors="black",
        linewidths=2,
        label="Target End",
        zorder=5,
    )

    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.title(
        f"Episode {episode}: Trajectories and Predictions (XY Plane)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_{episode}_trajectories.png", dpi=150)
    plt.close()

    # Plot 2: Agent actions over time
    plt.figure(figsize=(14, 4))
    plt.plot(time_steps_actions, actions_taken, "o-", markersize=3, linewidth=1)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Action (0-4)", fontsize=12)
    plt.title(
        f"Episode {episode}: Agent Actions Over Time", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.yticks([0, 1, 2, 3, 4], ["Left Max", "Left", "Straight", "Right", "Right Max"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_{episode}_actions.png", dpi=150)
    plt.close()

    # Plot 3: Range measurements vs real ranges
    plt.figure(figsize=(14, 5))
    plt.plot(time_steps, real_ranges, "b-", label="Real Range", linewidth=2)
    measured_indices = np.where(measured_ranges > 0)[0]
    if len(measured_indices) > 0:
        plt.scatter(
            time_steps[measured_indices],
            measured_ranges[measured_indices],
            c="red",
            s=30,
            label="Measured Range",
            alpha=0.7,
            zorder=5,
        )
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Range (m)", fontsize=12)
    plt.title(f"Episode {episode}: Range Measurements", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_{episode}_ranges.png", dpi=150)
    plt.close()

    # Plot 4: Velocities over time
    plt.figure(figsize=(14, 5))
    plt.plot(
        time_steps, agent_velocities[:-1], "b-", label="Agent Velocity", linewidth=2
    )
    plt.plot(
        time_steps, target_velocities[:-1], "r-", label="Target Velocity", linewidth=2
    )
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Velocity (m/s)", fontsize=12)
    plt.title(
        f"Episode {episode}: Velocities Over Time", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_{episode}_velocities.png", dpi=150)
    plt.close()

    # Plot 5: Tracking error and distance over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(time_steps, tracking_errors, "g-", linewidth=2)
    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Tracking Error (m)", fontsize=12)
    ax1.set_title("Tracking Error Over Time", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        y=np.mean(tracking_errors),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(tracking_errors):.2f} m",
        linewidth=2,
    )
    ax1.legend(fontsize=10)

    ax2.plot(time_steps, distances, "purple", linewidth=2)
    ax2.set_xlabel("Time (minutes)", fontsize=12)
    ax2.set_ylabel("Distance (m)", fontsize=12)
    ax2.set_title("Agent-Target Distance Over Time", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(
        y=np.mean(distances),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(distances):.2f} m",
        linewidth=2,
    )
    ax2.legend(fontsize=10)

    plt.suptitle(
        f"Episode {episode}: Tracking Performance",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_{episode}_performance.png", dpi=150)
    plt.close()

    print(f"Generated 5 plots for episode {episode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test simplified agent controller")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/17november/mappo_transformer_noisy_more_linear_utracking_1_vs_1_step456_rng1948878966.safetensors",
        help="Path to the model checkpoint",
    )
    args = parser.parse_args()

    test(model_path=args.model_path)
