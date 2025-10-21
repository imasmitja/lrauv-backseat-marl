import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from jaxagent.jax_agent import load_params
import json


def main():

    params = load_params(
        "IROS_MODELS/mappo_transformer_from_5v5follow_256steps_utracking_5_vs_5_step7320_rng202567368.safetensors"
    )
    d = jax.tree_util.tree_map(lambda x: x.shape, params["actor"])
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
