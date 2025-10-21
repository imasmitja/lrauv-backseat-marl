import torch
import torch.nn as nn
import numpy as np
from typing import Union, Dict, List, Optional
import math
import os

from .torch_modules import PPOActorTransformer
from .torch_utils import load_jax_params_to_torch


class CentralizedActorRNN:
    """PyTorch equivalent of the JAX CentralizedActorRNN"""

    def __init__(
        self,
        seed: int,
        agent_params_path: str,
        agent_list: List[str],
        landmark_list: List[str],
        actors_list: Optional[List[str]] = None,
        action_dim: int = 5,
        hidden_dim: int = 128,
        num_envs: int = 1,
        pos_norm: float = 1e-3,
        matrix_obs: bool = False,
        add_agent_id: bool = False,
        mask_ranges: bool = False,
        agent_class: str = "ppo_transformer",
        device: str = "cpu",
        **agent_kwargs,
    ):
        self.agent_list = agent_list
        self.landmark_list = landmark_list
        self.actors_list = agent_list if actors_list is None else actors_list
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.pos_norm = pos_norm
        self.matrix_obs = matrix_obs
        self.num_envs = num_envs
        self.add_agent_id = add_agent_id
        self.ranges_mask = 0.0 if mask_ranges else 1.0
        self.device = torch.device(device)

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Calculate input dimension based on observation structure
        self.obs_size = 6  # [x, y, z, range/rph_z, is_agent, is_self]
        self.total_obs_size = (
            len(self.agent_list) + len(self.landmark_list)
        ) * self.obs_size
        if self.add_agent_id:
            self.total_obs_size += len(self.agent_list)

        # Initialize the actor model
        if agent_class == "ppo_transformer":
            self.actor = PPOActorTransformer(
                input_dim=self.total_obs_size,
                action_dim=action_dim,
                hidden_size=hidden_dim,
                matrix_obs=matrix_obs,
                **agent_kwargs,
            ).to(self.device)
        else:
            raise ValueError(f"Invalid agent class: {agent_class}")

        # Load parameters from safetensors file
        if os.path.exists(agent_params_path):
            self.actor = load_jax_params_to_torch(self.actor, agent_params_path)
            print(f"Loaded parameters from {agent_params_path}")
        else:
            print(
                f"Warning: Parameter file {agent_params_path} not found. Using random initialization."
            )

        # Set to evaluation mode
        self.actor.eval()

        # Initialize hidden state
        self.reset(seed)

    def reset(self, seed: Optional[int] = None):
        """Reset hidden states and optionally the random seed"""
        self.hidden = torch.zeros(
            (len(self.actors_list), self.hidden_dim), device=self.device
        )
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def step(
        self, obs: Dict[str, Dict], done: bool, avail_actions: Optional[Dict] = None
    ) -> Dict[str, int]:
        """
        Take a step with the agent

        Args:
            obs: Dictionary of observations for each agent
            done: Boolean indicating if episode is done
            avail_actions: Dictionary of available actions for each agent

        Returns:
            Dictionary of actions for each actor
        """
        with torch.no_grad():
            # Process available actions
            if avail_actions is None:
                avail_actions_tensor = torch.ones(
                    (len(self.actors_list), self.action_dim),
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                avail_actions_list = []
                for actor in self.actors_list:
                    if actor in avail_actions:
                        avail_actions_list.append(
                            torch.tensor(avail_actions[actor], dtype=torch.bool)
                        )
                    else:
                        avail_actions_list.append(
                            torch.ones(self.action_dim, dtype=torch.bool)
                        )
                avail_actions_tensor = torch.stack(avail_actions_list).to(self.device)

            # Create done tensor
            dones = torch.full(
                (len(self.actors_list),), done, dtype=torch.bool, device=self.device
            )

            # Preprocess observations
            processed_obs = {}
            for agent, o in obs.items():
                if agent in self.actors_list:
                    processed_obs[agent] = self.preprocess_obs(agent, o)

            # Batchify observations
            if self.matrix_obs:
                obs_tensor = self.batchify_transformer(
                    processed_obs, self.actors_list, len(self.actors_list)
                )
            else:
                obs_tensor = self.batchify(
                    processed_obs, self.actors_list, len(self.actors_list)
                )

            obs_tensor = obs_tensor.to(self.device)

            # Add batch dimension to match JAX implementation
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
            dones = dones.unsqueeze(0)  # Add batch dimension

            # Forward pass through actor
            self.hidden, logits = self.actor(
                self.hidden, obs_tensor, dones, avail_actions_tensor.float()
            )

            # Get actions
            actions = torch.argmax(logits, dim=-1)

            # Remove the extra batch dimension that was added
            actions = actions.squeeze(0)

            # Unbatchify actions
            action_dict = self.unbatchify(
                actions, self.actors_list, self.num_envs, len(self.actors_list)
            )

            # Convert to int
            action_dict = {k: int(v.item()) for k, v in action_dict.items()}

            return action_dict

    def preprocess_obs(self, agent_name: str, obs: Dict) -> torch.Tensor:
        """
        Preprocess observations for a single agent

        Args:
            agent_name: Name of the agent
            obs: Raw observation dictionary

        Returns:
            Preprocessed observation tensor
        """
        # Pre-calculate sizes
        total_obs_size = self.total_obs_size

        # Preallocate observation array
        obs_array = torch.zeros(total_obs_size, device=self.device)

        # Index for filling the array
        idx = 0

        # Add other agents' observations
        for agent in self.agent_list:
            if agent == agent_name:
                # Add self observation
                self_obs = torch.tensor(
                    [
                        obs["x"] * self.pos_norm,
                        obs["y"] * self.pos_norm,
                        obs["z"] * self.pos_norm,
                        obs["rph_z"] * self.pos_norm,
                        1.0,  # is agent
                        1.0,  # is self
                    ],
                    device=self.device,
                )
                obs_array[idx : idx + self.obs_size] = self_obs
                idx += self.obs_size
            else:
                # Add other agent observation
                other_obs = torch.tensor(
                    [
                        obs[f"{agent}_dx"] * self.pos_norm,
                        obs[f"{agent}_dy"] * self.pos_norm,
                        obs[f"{agent}_dz"] * self.pos_norm,
                        math.sqrt(
                            obs[f"{agent}_dx"] ** 2
                            + obs[f"{agent}_dy"] ** 2
                            + obs[f"{agent}_dz"] ** 2
                        )
                        * self.pos_norm
                        * self.ranges_mask,
                        1.0,  # is agent
                        0.0,  # is self
                    ],
                    device=self.device,
                )
                obs_array[idx : idx + self.obs_size] = other_obs
                idx += self.obs_size

        # Add landmarks' observations
        for landmark in self.landmark_list:
            landmark_obs = torch.tensor(
                [
                    (obs["x"] - obs[f"{landmark}_tracking_x"]) * self.pos_norm,
                    (obs["y"] - obs[f"{landmark}_tracking_y"]) * self.pos_norm,
                    (obs["z"] - obs[f"{landmark}_tracking_z"]) * self.pos_norm,
                    obs[f"{landmark}_range"] * self.pos_norm * self.ranges_mask,
                    0.0,  # is agent
                    0.0,  # is self
                ],
                device=self.device,
            )
            obs_array[idx : idx + self.obs_size] = landmark_obs
            idx += self.obs_size

        # Add agent id
        if self.add_agent_id:
            agent_id = torch.tensor(
                [1.0 if agent == agent_name else 0.0 for agent in self.agent_list],
                device=self.device,
            )
            obs_array[idx : idx + len(self.agent_list)] = agent_id

        if self.matrix_obs:
            obs_array = obs_array.reshape(
                (len(self.agent_list) + len(self.landmark_list), self.obs_size)
            )

        # Check for NaN values
        if torch.isnan(obs_array).any():
            raise ValueError(f"NaN in the observation of agent {agent_name}")

        return obs_array

    def batchify(
        self, x: Dict[str, torch.Tensor], agent_list: List[str], num_actors: int
    ) -> torch.Tensor:
        """Batchify observations for vector input"""
        tensors = [x[a] for a in agent_list]
        stacked = torch.stack(tensors)
        return stacked.reshape((num_actors, -1))

    def batchify_transformer(
        self, x: Dict[str, torch.Tensor], agent_list: List[str], num_actors: int
    ) -> torch.Tensor:
        """Batchify observations for transformer input (keeping entity dimension)"""
        tensors = [x[a] for a in agent_list]
        stacked = torch.stack(tensors)
        num_entities = stacked.shape[-2]
        num_feats = stacked.shape[-1]
        return stacked.reshape((num_actors, num_entities, num_feats))

    def unbatchify(
        self, x: torch.Tensor, agent_list: List[str], num_envs: int, num_actors: int
    ) -> Dict[str, torch.Tensor]:
        """Unbatchify actions back to agent dictionary"""
        x = x.reshape((num_actors, num_envs, -1))
        return {a: x[i] for i, a in enumerate(agent_list)}

    def get_hidden_state(self) -> torch.Tensor:
        """Get current hidden state"""
        return self.hidden.clone()

    def set_hidden_state(self, hidden_state: torch.Tensor):
        """Set hidden state"""
        self.hidden = hidden_state.to(self.device)

    def save_model(self, filepath: str):
        """Save the PyTorch model"""
        torch.save(self.actor.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a PyTorch model"""
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device))
        self.actor.eval()
        print(f"Model loaded from {filepath}")


def create_torch_agent_from_jax_params(
    safetensors_path: str,
    agent_list: List[str],
    landmark_list: List[str],
    seed: int = 0,
    **kwargs,
) -> CentralizedActorRNN:
    """
    Create a PyTorch agent by loading parameters from a JAX safetensors file

    Args:
        safetensors_path: Path to the JAX safetensors file
        agent_list: List of agent names
        landmark_list: List of landmark names
        seed: Random seed
        **kwargs: Additional arguments for the agent

    Returns:
        Initialized PyTorch agent
    """
    return CentralizedActorRNN(
        seed=seed,
        agent_params_path=safetensors_path,
        agent_list=agent_list,
        landmark_list=landmark_list,
        **kwargs,
    )
