import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Dict
import math
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

from .jax_modules import PPOActorTransformer


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_transformer(x: dict, agent_list, num_actors):
    # bathify specifically for transformer keeping the last two dimensions (entities, features)
    x = jnp.stack([x[a] for a in agent_list])
    num_entities = x.shape[-2]
    num_feats = x.shape[-1]
    x = x.reshape((num_actors, num_entities, num_feats))
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class CentralizedActorRNN:

    def __init__(
        self,
        seed,
        agent_params,
        agent_list,
        landmark_list,
        actors_list=None,  # actual actors to control, if None all agents
        action_dim=5,
        hidden_dim=128,
        num_envs=1,
        pos_norm=1e-3,
        matrix_obs=False,
        add_agent_id=False,
        mask_ranges=False,
        agent_class=True,
        **agent_kwargs,
    ):

        self.agent_params = agent_params
        if "params" not in self.agent_params:
            self.agent_params = {"params": self.agent_params}
        self.agent_list = agent_list
        self.landmark_list = landmark_list
        self.actors_list = agent_list if actors_list is None else actors_list
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.pos_norm = pos_norm
        self.matrix_obs = matrix_obs
        self.num_envs = num_envs
        self.rng = jax.random.PRNGKey(seed)
        self.add_agent_id = add_agent_id
        self.ranges_mask = 0.0 if mask_ranges else 1.0

        if agent_class == "ppo_transformer":
            agent_class = PPOActorTransformer
        else:
            raise ValueError(f"Invalid agent class: {agent_class}")

        self.actor = agent_class(action_dim, hidden_dim, **agent_kwargs)

        self.jitted_actor_apply = jax.jit(self.actor.apply)

    def reset(self, seed=None):
        self.hidden = jnp.zeros((len(self.actors_list), self.hidden_dim))
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng)

    def step(self, obs, done, avail_actions=None):

        if avail_actions is None:
            avail_actions = jnp.ones(
                (len(self.actors_list), self.action_dim), dtype=bool
            )
        else:
            avail_actions = {k: jnp.array(v) for k, v in avail_actions.items()}
            avail_actions = batchify(
                avail_actions, self.actors_list, len(self.actors_list)
            )

        # assumes a single boolean done signal
        dones = jnp.full((len(self.actors_list),), done, dtype=bool)

        self.rng, key = jax.random.split(self.rng)
        obs = {agent: self.preprocess_obs(agent, o) for agent, o in obs.items()}

        if self.matrix_obs:
            obs = batchify_transformer(obs, self.actors_list, len(self.actors_list))
        else:
            obs = batchify(obs, self.actors_list, len(self.actors_list))

        ac_in = (
            obs[np.newaxis, ...],
            dones[np.newaxis, ...],
            avail_actions,
        )

        self.hidden, logits = self.jitted_actor_apply(
            self.agent_params, self.hidden, ac_in
        )
        action = jnp.argmax(logits, axis=-1)

        action = unbatchify(
            action, self.actors_list, self.num_envs, len(self.actors_list)
        )
        action = {k: int(v.squeeze()) for k, v in action.items()}

        return action

    def preprocess_obs(self, agent_name, obs):

        # Pre-calculate sizes
        obs_size = 6
        total_obs_size = (len(self.agent_list) + len(self.landmark_list)) * obs_size

        if self.add_agent_id:
            total_obs_size += len(self.agent_list)

        # Preallocate observation array
        obs_array = jnp.zeros(total_obs_size)

        # Index for filling the array
        idx = 0

        # Add other agents' observations
        for agent in self.agent_list:
            if agent == agent_name:
                # Add self observation
                obs_array = obs_array.at[idx : idx + obs_size].set(
                    [
                        obs["x"] * self.pos_norm,
                        obs["y"] * self.pos_norm,
                        obs["z"] * self.pos_norm,
                        obs["rph_z"] * self.pos_norm,
                        1,  # is agent
                        1,  # is self
                    ]
                )
                idx += obs_size

            else:
                obs_array = obs_array.at[idx : idx + obs_size].set(
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
                        1,  # is agent
                        0,  # is self
                    ]
                )
                idx += obs_size

        # Add landmarks' observations
        for landmark in self.landmark_list:
            obs_array = obs_array.at[idx : idx + obs_size].set(
                [
                    (obs["x"] - obs[f"{landmark}_tracking_x"]) * self.pos_norm,
                    (obs["y"] - obs[f"{landmark}_tracking_y"]) * self.pos_norm,
                    (obs["z"] - obs[f"{landmark}_tracking_z"]) * self.pos_norm,
                    obs[f"{landmark}_range"] * self.pos_norm * self.ranges_mask,
                    0,  # is agent
                    0,  # is self
                ]
            )
            idx += obs_size

        # Add agent id
        if self.add_agent_id:
            obs_array = obs_array.at[idx : idx + len(self.agent_list)].set(
                [1 if agent == agent_name else 0 for agent in self.agent_list]
            )

        if self.matrix_obs:
            obs_array = obs_array.reshape(
                (len(self.agent_list) + len(self.landmark_list), obs_size)
            )

        # Check for NaN values
        if jnp.isnan(obs_array).any():
            raise ValueError(f"NaN in the observation of agent {agent_name}")

        return obs_array
