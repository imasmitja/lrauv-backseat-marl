#!/usr/bin/env python3
"""
Test script to validate that PyTorch and JAX implementations produce identical outputs.
Tests both single-step inference and recurrent behavior over trajectories.
"""

import os
import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any

# Add the jaxagent package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxagent.jax_agent import CentralizedActorRNN as JAXCentralizedActorRNN
from jaxagent.torch_agent import CentralizedActorRNN as TorchCentralizedActorRNN
from jaxagent.jax_agent import load_params


def create_sample_observation(
    agent_list: List[str], landmark_list: List[str]
) -> Dict[str, Dict[str, float]]:
    """Create sample observations for testing"""
    obs = {}

    for i, agent in enumerate(agent_list):
        agent_obs = {
            "x": float(i * 2.0),
            "y": float(i * 1.5),
            "z": float(i * 0.5),
            "rph_z": float(i * 0.1),
        }

        # Add relative positions to other agents
        for j, other_agent in enumerate(agent_list):
            if other_agent != agent:
                agent_obs[f"{other_agent}_dx"] = float((j - i) * 1.0)
                agent_obs[f"{other_agent}_dy"] = float((j - i) * 0.8)
                agent_obs[f"{other_agent}_dz"] = float((j - i) * 0.2)

        # Add landmark tracking information
        for k, landmark in enumerate(landmark_list):
            agent_obs[f"{landmark}_tracking_x"] = float(k * 3.0)
            agent_obs[f"{landmark}_tracking_y"] = float(k * 2.0)
            agent_obs[f"{landmark}_tracking_z"] = float(k * 1.0)
            agent_obs[f"{landmark}_range"] = float(k * 0.5 + 1.0)

        obs[agent] = agent_obs

    return obs


def create_random_observation_sequence(
    agent_list: List[str],
    landmark_list: List[str],
    sequence_length: int,
    seed: int = 42,
) -> List[Tuple[Dict[str, Dict[str, float]], bool]]:
    """Create a sequence of random observations for trajectory testing"""
    np.random.seed(seed)
    observations = []

    for step in range(sequence_length):
        obs = {}

        for i, agent in enumerate(agent_list):
            agent_obs = {
                "x": np.random.uniform(-10, 10),
                "y": np.random.uniform(-10, 10),
                "z": np.random.uniform(-5, 5),
                "rph_z": np.random.uniform(-np.pi, np.pi),
            }

            # Add relative positions to other agents
            for j, other_agent in enumerate(agent_list):
                if other_agent != agent:
                    agent_obs[f"{other_agent}_dx"] = np.random.uniform(-5, 5)
                    agent_obs[f"{other_agent}_dy"] = np.random.uniform(-5, 5)
                    agent_obs[f"{other_agent}_dz"] = np.random.uniform(-2, 2)

            # Add landmark tracking information
            for k, landmark in enumerate(landmark_list):
                agent_obs[f"{landmark}_tracking_x"] = np.random.uniform(-15, 15)
                agent_obs[f"{landmark}_tracking_y"] = np.random.uniform(-15, 15)
                agent_obs[f"{landmark}_tracking_z"] = np.random.uniform(-8, 8)
                agent_obs[f"{landmark}_range"] = np.random.uniform(0.1, 20.0)

            obs[agent] = agent_obs

        # Episode done signal (done at the end for testing reset)
        done = step == sequence_length - 1
        observations.append((obs, done))

    return observations


def test_single_step_equivalence(
    jax_agent: JAXCentralizedActorRNN,
    torch_agent: TorchCentralizedActorRNN,
    agent_list: List[str],
    landmark_list: List[str],
    tolerance: float = 1e-5,
) -> bool:
    """Test that both agents produce the same output for a single step"""
    print("Testing single-step equivalence...")

    # Create sample observation
    obs = create_sample_observation(agent_list, landmark_list)
    done = False

    # Reset both agents with the same seed
    jax_agent.reset(seed=42)
    torch_agent.reset(seed=42)

    # Get actions from both agents
    jax_actions = jax_agent.step(obs, done)
    torch_actions = torch_agent.step(obs, done)

    print(f"JAX actions: {jax_actions}")
    print(f"PyTorch actions: {torch_actions}")

    # Compare actions
    success = True
    for agent in agent_list:
        if agent in jax_actions and agent in torch_actions:
            if jax_actions[agent] != torch_actions[agent]:
                print(
                    f"Action mismatch for {agent}: JAX={jax_actions[agent]}, PyTorch={torch_actions[agent]}"
                )
                success = False
        else:
            print(f"Missing agent {agent} in one of the action dictionaries")
            success = False

    if success:
        print("✓ Single-step test PASSED")
    else:
        print("✗ Single-step test FAILED")

    return success


def test_hidden_state_consistency(
    jax_agent: JAXCentralizedActorRNN,
    torch_agent: TorchCentralizedActorRNN,
    agent_list: List[str],
    landmark_list: List[str],
    tolerance: float = 1e-4,
) -> bool:
    """Test that hidden states evolve consistently"""
    print("\nTesting hidden state consistency...")

    # Create sample observation
    obs = create_sample_observation(agent_list, landmark_list)

    # Reset both agents
    jax_agent.reset(seed=42)
    torch_agent.reset(seed=42)

    # Take a few steps and compare hidden states
    success = True
    for step in range(3):
        print(f"Step {step + 1}:")

        # Take step
        jax_actions = jax_agent.step(obs, False)
        torch_actions = torch_agent.step(obs, False)

        # Get hidden states (assuming we can access them)
        jax_hidden = jax_agent.hidden  # JAX hidden state
        torch_hidden = torch_agent.get_hidden_state()  # PyTorch hidden state

        # Convert to numpy for comparison
        jax_hidden_np = np.array(jax_hidden)
        torch_hidden_np = torch_hidden.cpu().numpy()

        # Compare shapes
        if jax_hidden_np.shape != torch_hidden_np.shape:
            print(
                f"  Hidden state shape mismatch: JAX={jax_hidden_np.shape}, PyTorch={torch_hidden_np.shape}"
            )
            success = False
            continue

        # Compare values
        diff = np.abs(jax_hidden_np - torch_hidden_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"  Hidden state max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")

        if max_diff > tolerance:
            print(f"  ✗ Hidden state difference too large at step {step + 1}")
            success = False
        else:
            print(f"  ✓ Hidden states match within tolerance")

        # Modify observation slightly for next step
        for agent in obs:
            obs[agent]["x"] += 0.1
            obs[agent]["y"] += 0.05

    if success:
        print("✓ Hidden state consistency test PASSED")
    else:
        print("✗ Hidden state consistency test FAILED")

    return success


def test_trajectory_recurrency(
    jax_agent: JAXCentralizedActorRNN,
    torch_agent: TorchCentralizedActorRNN,
    agent_list: List[str],
    landmark_list: List[str],
    sequence_length: int = 10,
    num_trials: int = 3,
) -> bool:
    """Test that both agents produce identical action sequences for the same trajectory"""
    print(
        f"\nTesting trajectory recurrency over {sequence_length} steps, {num_trials} trials..."
    )

    success = True

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}:")

        # Create random observation sequence
        obs_sequence = create_random_observation_sequence(
            agent_list, landmark_list, sequence_length, seed=trial + 100
        )

        # Reset both agents with the same seed
        jax_agent.reset(seed=trial + 200)
        torch_agent.reset(seed=trial + 200)

        jax_action_sequence = []
        torch_action_sequence = []

        # Run through the entire sequence
        for step, (obs, done) in enumerate(obs_sequence):
            jax_actions = jax_agent.step(obs, done)
            torch_actions = torch_agent.step(obs, done)

            jax_action_sequence.append(jax_actions)
            torch_action_sequence.append(torch_actions)

            # Compare actions at each step
            step_match = True
            for agent in agent_list:
                if jax_actions.get(agent) != torch_actions.get(agent):
                    print(
                        f"  Step {step}: Action mismatch for {agent}: "
                        f"JAX={jax_actions.get(agent)}, PyTorch={torch_actions.get(agent)}"
                    )
                    step_match = False
                    success = False

            if step_match:
                print(f"  Step {step}: ✓ Actions match")
            else:
                print(f"  Step {step}: ✗ Actions differ")

        if success:
            print(f"Trial {trial + 1}: ✓ PASSED")
        else:
            print(f"Trial {trial + 1}: ✗ FAILED")

    if success:
        print("✓ Trajectory recurrency test PASSED")
    else:
        print("✗ Trajectory recurrency test FAILED")

    return success


def test_reset_behavior(
    jax_agent: JAXCentralizedActorRNN,
    torch_agent: TorchCentralizedActorRNN,
    agent_list: List[str],
    landmark_list: List[str],
) -> bool:
    """Test that reset behavior is consistent between implementations"""
    print("\nTesting reset behavior...")

    obs = create_sample_observation(agent_list, landmark_list)

    # Take some steps to change hidden state
    jax_agent.reset(seed=42)
    torch_agent.reset(seed=42)

    for _ in range(3):
        jax_agent.step(obs, False)
        torch_agent.step(obs, False)

    # Now reset both and take a step
    jax_agent.reset(seed=42)
    torch_agent.reset(seed=42)

    jax_actions_after_reset = jax_agent.step(obs, False)
    torch_actions_after_reset = torch_agent.step(obs, False)

    # Reset again and take the same step - should get the same result
    jax_agent.reset(seed=42)
    torch_agent.reset(seed=42)

    jax_actions_repeat = jax_agent.step(obs, False)
    torch_actions_repeat = torch_agent.step(obs, False)

    # Check consistency
    jax_consistent = jax_actions_after_reset == jax_actions_repeat
    torch_consistent = torch_actions_after_reset == torch_actions_repeat
    cross_consistent = jax_actions_after_reset == torch_actions_after_reset

    success = jax_consistent and torch_consistent and cross_consistent

    print(f"JAX reset consistency: {jax_consistent}")
    print(f"PyTorch reset consistency: {torch_consistent}")
    print(f"Cross-implementation consistency: {cross_consistent}")

    if success:
        print("✓ Reset behavior test PASSED")
    else:
        print("✗ Reset behavior test FAILED")

    return success


def main():
    """Main testing function"""
    print("=== JAX vs PyTorch Agent Equivalence Tests ===\n")

    # Configuration
    agent_list = ["agent_0", "agent_1"]
    landmark_list = ["landmark_0", "landmark_1"]
    action_dim = 5
    hidden_dim = 64

    # Find a safetensors file to test with
    model_dir = "/Users/cotechino/x/jaxagent/IROS_MODELS"
    safetensors_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

    if not safetensors_files:
        print("No safetensors files found in IROS_MODELS directory!")
        return False

    # Use the first available safetensors file
    safetensors_path = os.path.join(model_dir, safetensors_files[0])
    print(f"Using model: {safetensors_files[0]}")
    print(f"Full path: {safetensors_path}\n")

    try:
        # Load JAX agent
        print("Loading JAX agent...")
        jax_params = load_params(safetensors_path)["actor"]
        jax_agent = JAXCentralizedActorRNN(
            seed=42,
            agent_params=jax_params,
            agent_list=agent_list,
            landmark_list=landmark_list,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            matrix_obs=True,
            agent_class="ppo_transformer",
        )
        print("✓ JAX agent loaded successfully")

        # Load PyTorch agent
        print("Loading PyTorch agent...")
        torch_agent = TorchCentralizedActorRNN(
            seed=42,
            agent_params_path=safetensors_path,
            agent_list=agent_list,
            landmark_list=landmark_list,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            matrix_obs=True,
            agent_class="ppo_transformer",
            device="cpu",
        )
        print("✓ PyTorch agent loaded successfully")

    except Exception as e:
        print(f"Error loading agents: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Run tests
    all_tests_passed = True

    tests = [
        (test_single_step_equivalence, "Single Step Equivalence"),
        (test_hidden_state_consistency, "Hidden State Consistency"),
        (test_trajectory_recurrency, "Trajectory Recurrency"),
        (test_reset_behavior, "Reset Behavior"),
    ]

    for test_func, test_name in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print("=" * 50)

            if test_func == test_trajectory_recurrency:
                result = test_func(
                    jax_agent,
                    torch_agent,
                    agent_list,
                    landmark_list,
                    sequence_length=5,
                    num_trials=2,
                )
            else:
                result = test_func(jax_agent, torch_agent, agent_list, landmark_list)

            if not result:
                all_tests_passed = False

        except Exception as e:
            print(f"Error in {test_name}: {e}")
            import traceback

            traceback.print_exc()
            all_tests_passed = False

    # Final result
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print("=" * 50)

    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("JAX and PyTorch implementations are equivalent.")
    else:
        print("❌ SOME TESTS FAILED")
        print("There are differences between JAX and PyTorch implementations.")

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
