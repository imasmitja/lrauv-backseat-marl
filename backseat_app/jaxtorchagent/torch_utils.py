import torch
import numpy as np
from safetensors import safe_open
from typing import Dict, Any
import os


def load_jax_params_to_torch(
    model: torch.nn.Module, safetensors_path: str
) -> torch.nn.Module:
    """
    Load JAX/Flax parameters from safetensors file and convert them to PyTorch format.

    Args:
        model: PyTorch model to load parameters into
        safetensors_path: Path to the safetensors file containing JAX parameters

    Returns:
        Model with loaded parameters
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    # Load the safetensors file
    jax_params = {}
    with safe_open(safetensors_path, framework="numpy") as f:
        for key in f.keys():
            if "actor" in key:
                jax_params[key] = f.get_tensor(key)

    print(jax_params)

    # Convert JAX parameter structure to PyTorch
    torch_state_dict = convert_jax_to_torch_params(jax_params, model)

    # Load into the model
    missing_keys, unexpected_keys = model.load_state_dict(
        torch_state_dict, strict=False
    )

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print(f"Successfully loaded {len(torch_state_dict)} parameters")

    return model


def convert_jax_to_torch_params(
    jax_params: Dict[str, np.ndarray], model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Convert JAX/Flax parameter structure to PyTorch state dict format.

    This function handles the mapping between JAX/Flax naming conventions and PyTorch naming.
    """
    torch_state_dict = {}

    # Get the model's state dict for reference
    model_state_dict = model.state_dict()

    # Print available JAX parameters for debugging
    print("Available JAX parameters:")
    for key in sorted(jax_params.keys()):
        print(f"  {key}: {jax_params[key].shape}")

    print("\nExpected PyTorch parameters:")
    for key in sorted(model_state_dict.keys()):
        print(f"  {key}: {model_state_dict[key].shape}")

    print("\nParameter mapping:")

    # Track which PyTorch parameters we've successfully mapped
    mapped_params = set()

    # Mapping logic based on the actual parameter structure
    for jax_key, jax_param in jax_params.items():
        torch_key = map_jax_key_to_torch(jax_key)

        if torch_key and torch_key in model_state_dict:
            # Convert numpy array to torch tensor
            torch_param = torch.from_numpy(jax_param).float()

            # Handle parameter shape transformations if needed
            torch_param = adjust_parameter_shape(
                torch_param, jax_key, torch_key, model_state_dict[torch_key].shape
            )

            torch_state_dict[torch_key] = torch_param
            mapped_params.add(torch_key)
            print(f"✓ Mapped: {jax_key} -> {torch_key} (shape: {torch_param.shape})")
        else:
            if torch_key:
                print(f"✗ PyTorch key not found: {jax_key} -> {torch_key}")
            else:
                print(f"? Unmapped JAX parameter: {jax_key}")

    # Check for unmapped PyTorch parameters
    unmapped_torch_params = set(model_state_dict.keys()) - mapped_params
    if unmapped_torch_params:
        print(f"\nWarning: Unmapped PyTorch parameters:")
        for param in sorted(unmapped_torch_params):
            print(f"  {param}: {model_state_dict[param].shape}")

    print(f"\nMapped {len(mapped_params)}/{len(model_state_dict)} PyTorch parameters")

    return torch_state_dict


def map_jax_key_to_torch(jax_key: str) -> str:
    """
    Map JAX/Flax parameter names to PyTorch parameter names.

    Based on the actual JAX parameter structure:
    - actor,params,Dense_0 -> action_head
    - actor,params,Embedder_0,Dense_0 -> embedder
    - actor,params,Embedder_0,LayerNorm_0 -> embedder layer norm (not used in current arch)
    - actor,params,ScannedTransformer_0,encoders_X -> transformer.encoders.X
    """
    # Remove the 'actor,params,' prefix if present (common in Flax)
    if jax_key.startswith("actor,params,"):
        jax_key = jax_key[13:]  # Remove 'actor,params,'
    elif jax_key.startswith("actor,"):
        jax_key = jax_key[6:]  # Remove 'actor,'
    elif jax_key.startswith("params,"):
        jax_key = jax_key[7:]  # Remove 'params,'

    # Handle top-level Dense layer (action head)
    if jax_key.startswith("Dense_0,"):
        if "kernel" in jax_key:
            return "action_head.weight"
        elif "bias" in jax_key:
            return "action_head.bias"

    # Handle Embedder
    if jax_key.startswith("Embedder_0,"):
        if "Dense_0,kernel" in jax_key:
            return "embedder.weight"
        elif "Dense_0,bias" in jax_key:
            return "embedder.bias"
        elif "LayerNorm_0,scale" in jax_key:
            return "embedder_norm.weight"
        elif "LayerNorm_0,bias" in jax_key:
            return "embedder_norm.bias"

    # Handle ScannedTransformer
    if jax_key.startswith("ScannedTransformer_0,"):
        # Extract layer number from encoders_X
        if "encoders_" in jax_key:
            # Split and find the encoder layer index
            parts = jax_key.split(",")
            layer_idx = None
            for part in parts:
                if part.startswith("encoders_"):
                    layer_idx = part.split("_")[1]
                    break

            if layer_idx is not None:
                # Map self-attention components
                if "self_attn" in jax_key:
                    if "query,kernel" in jax_key:
                        return (
                            f"transformer.encoders.{layer_idx}.self_attn.q_proj.weight"
                        )
                    elif "key,kernel" in jax_key:
                        return (
                            f"transformer.encoders.{layer_idx}.self_attn.k_proj.weight"
                        )
                    elif "value,kernel" in jax_key:
                        return (
                            f"transformer.encoders.{layer_idx}.self_attn.v_proj.weight"
                        )
                    elif "out,kernel" in jax_key:
                        return f"transformer.encoders.{layer_idx}.self_attn.out_proj.weight"

                # Map feed-forward components
                elif "linear_0,kernel" in jax_key:
                    return f"transformer.encoders.{layer_idx}.linear1.weight"
                elif "linear_0,bias" in jax_key:
                    return f"transformer.encoders.{layer_idx}.linear1.bias"
                elif "linear_1,kernel" in jax_key:
                    return f"transformer.encoders.{layer_idx}.linear2.weight"
                elif "linear_1,bias" in jax_key:
                    return f"transformer.encoders.{layer_idx}.linear2.bias"

                # Map layer normalization components
                elif "norm1,scale" in jax_key:
                    return f"transformer.encoders.{layer_idx}.norm1.weight"
                elif "norm1,bias" in jax_key:
                    return f"transformer.encoders.{layer_idx}.norm1.bias"
                elif "norm2,scale" in jax_key:
                    return f"transformer.encoders.{layer_idx}.norm2.weight"
                elif "norm2,bias" in jax_key:
                    return f"transformer.encoders.{layer_idx}.norm2.bias"

    return None


def adjust_parameter_shape(
    torch_param: torch.Tensor, jax_key: str, torch_key: str, expected_shape: torch.Size
) -> torch.Tensor:
    """
    Adjust parameter shapes if needed (e.g., transpose weight matrices).

    JAX/Flax and PyTorch may use different conventions for weight matrix shapes.
    Based on the actual JAX structure:
    - Regular linear layers: JAX (in_features, out_features) -> PyTorch (out_features, in_features)
    - Attention weights: JAX (input_dim, num_heads, head_dim) -> PyTorch (output_dim, input_dim)
    - Attention out: JAX (num_heads, head_dim, output_dim) -> PyTorch (output_dim, input_dim)
    """

    # Handle self-attention weights which are 3D in JAX
    if "self_attn" in torch_key and torch_param.dim() == 3:
        if "q_proj" in torch_key or "k_proj" in torch_key or "v_proj" in torch_key:
            # JAX: (input_dim, num_heads, head_dim) -> PyTorch: (output_dim, input_dim)
            # output_dim = num_heads * head_dim
            input_dim, num_heads, head_dim = torch_param.shape
            output_dim = num_heads * head_dim

            # Reshape to (input_dim, output_dim) then transpose to (output_dim, input_dim)
            reshaped = torch_param.reshape(input_dim, output_dim)
            transposed = reshaped.transpose(0, 1)

            if transposed.shape == expected_shape:
                print(
                    f"  Reshaped 3D attention weight {torch_param.shape} -> {transposed.shape}"
                )
                return transposed
            else:
                print(f"  Warning: 3D attention reshape failed for {torch_key}")
                print(
                    f"    JAX shape: {torch_param.shape}, Expected: {expected_shape}, Got: {transposed.shape}"
                )

        elif "out_proj" in torch_key:
            # JAX: (num_heads, head_dim, output_dim) -> PyTorch: (output_dim, input_dim)
            # input_dim = num_heads * head_dim
            num_heads, head_dim, output_dim = torch_param.shape
            input_dim = num_heads * head_dim

            # The correct transformation is:
            # JAX (8, 8, 64) -> permute(2, 0, 1) -> (64, 8, 8) -> reshape -> (64, 64)
            # This matches the inverse of the PyTorch operation in forward()
            permuted = torch_param.permute(2, 0, 1)  # (output_dim, num_heads, head_dim)
            final = permuted.reshape(output_dim, input_dim)  # (output_dim, input_dim)

            if final.shape == expected_shape:
                print(
                    f"  Reshaped 3D out projection {torch_param.shape} -> {final.shape}"
                )
                return final
            else:
                print(f"  Warning: 3D out projection reshape failed for {torch_key}")
                print(
                    f"    JAX shape: {torch_param.shape}, Expected: {expected_shape}, Got: {final.shape}"
                )

    # Handle regular 2D linear layer weights
    elif "kernel" in jax_key and "weight" in torch_key and torch_param.dim() == 2:
        # JAX: (in_features, out_features) -> PyTorch: (out_features, in_features)
        if torch_param.shape != expected_shape:
            transposed = torch_param.transpose(0, 1)
            if transposed.shape == expected_shape:
                print(
                    f"  Transposed 2D weight {torch_param.shape} -> {transposed.shape}"
                )
                return transposed
            else:
                print(f"  Warning: Transpose didn't fix shape mismatch for {torch_key}")
                print(
                    f"    JAX shape: {torch_param.shape}, Expected: {expected_shape}, Got: {transposed.shape}"
                )

    # Handle bias terms and layer norm parameters (should match directly)
    elif "bias" in jax_key and "bias" in torch_key:
        if torch_param.shape == expected_shape:
            return torch_param
    elif "scale" in jax_key and "weight" in torch_key:
        # LayerNorm scale -> weight
        if torch_param.shape == expected_shape:
            return torch_param

    # If we get here, there might be a shape mismatch
    if torch_param.shape != expected_shape:
        print(f"  Warning: Shape mismatch for {torch_key}")
        print(f"    JAX key: {jax_key}, JAX shape: {torch_param.shape}")
        print(f"    PyTorch key: {torch_key}, Expected shape: {expected_shape}")

        # Try to use strict=False loading by returning parameter as-is
        # This allows the model to load even with mismatched shapes
        print(f"    Returning parameter as-is for non-strict loading")

    return torch_param


def save_torch_params_to_safetensors(model: torch.nn.Module, filepath: str):
    """
    Save PyTorch model parameters to safetensors format.
    """
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    save_file(state_dict, filepath)


def compare_model_outputs(
    jax_model, torch_model, jax_params, sample_input, hidden_state=None, tolerance=1e-5
):
    """
    Compare outputs from JAX and PyTorch models to verify correct parameter loading.
    """
    import jax
    import jax.numpy as jnp

    # Convert inputs to appropriate formats
    if isinstance(sample_input, dict):
        # Handle dictionary inputs
        jax_input = {k: jnp.array(v) for k, v in sample_input.items()}
        torch_input = {
            k: torch.from_numpy(np.array(v)).float() for k, v in sample_input.items()
        }
    else:
        jax_input = jnp.array(sample_input)
        torch_input = torch.from_numpy(np.array(sample_input)).float()

    # JAX forward pass
    if hidden_state is not None:
        jax_hidden = jnp.array(hidden_state)
        jax_output = jax_model.apply(jax_params, jax_hidden, jax_input)
    else:
        jax_output = jax_model.apply(jax_params, jax_input)

    # PyTorch forward pass
    torch_model.eval()
    with torch.no_grad():
        if hidden_state is not None:
            torch_hidden = torch.from_numpy(np.array(hidden_state)).float()
            torch_output = torch_model(torch_hidden, torch_input)
        else:
            torch_output = torch_model(torch_input)

    # Compare outputs
    if isinstance(jax_output, tuple) and isinstance(torch_output, tuple):
        for i, (jax_out, torch_out) in enumerate(zip(jax_output, torch_output)):
            jax_np = np.array(jax_out)
            torch_np = torch_out.cpu().numpy()

            diff = np.abs(jax_np - torch_np)
            max_diff = np.max(diff)

            print(f"Output {i}: Max difference = {max_diff}")
            if max_diff > tolerance:
                print(f"Warning: Large difference detected in output {i}")
                return False
    else:
        jax_np = np.array(jax_output)
        torch_np = torch_output.cpu().numpy()

        diff = np.abs(jax_np - torch_np)
        max_diff = np.max(diff)

        print(f"Max difference = {max_diff}")
        if max_diff > tolerance:
            print("Warning: Large difference detected")
            return False

    print("Models produce similar outputs!")
    return True
