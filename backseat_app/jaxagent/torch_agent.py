import os
import torch
import numpy as np
from typing import Union, Dict, List, Optional
import math
from safetensors.torch import save_file, load_file

from .torch_modules import PPOActorRNN, PPOActorTransformer, PQNRnn


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    # For PyTorch, we can use torch.save or safetensors
    if filename.endswith('.safetensors'):
        save_file(params, filename)
    else:
        torch.save(params, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    if filename.endswith('.safetensors'):
        return load_file(filename)
    else:
        return torch.load(filename, map_location='cpu')


def batchify(x: dict, agent_list: List[str], num_actors: int) -> torch.Tensor:
    x = torch.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_transformer(x: dict, agent_list: List[str], num_actors: int) -> torch.Tensor:
    # batchify specifically for transformer keeping the last two dimensions (entities, features)
    x = torch.stack([x[a] for a in agent_list])
    num_entities = x.shape[-2]
    num_feats = x.shape[-1]
    x = x.reshape((num_actors, num_entities, num_feats))
    return x


def unbatchify(x: torch.Tensor, agent_list: List[str], num_envs: int, num_actors: int) -> Dict[str, torch.Tensor]:
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class CentralizedActorRNN:

    def __init__(
        self,
        seed: int,
        agent_params: Dict,
        agent_list: List[str],
        landmark_list: List[str],
        actors_list: Optional[List[str]] = None,  # actual actors to control, if None all agents
        action_dim: int = 5,
        hidden_dim: int = 128,
        num_envs: int = 1,
        pos_norm: float = 1e-3,
        matrix_obs: bool = False,
        add_agent_id: bool = False,
        mask_ranges: bool = False,
        agent_class: str = 'ppo_rnn',
        device: str = 'cpu',
        **agent_kwargs,
    ):
        
        self.device = device
        self.agent_params = agent_params
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

        # Set random seed
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

        # Initialize agent class
        if agent_class == 'ppo_rnn':
            agent_class_obj = PPOActorRNN
        elif agent_class == 'pqn_rnn':
            agent_class_obj = PQNRnn
        elif agent_class == 'ppo_transformer':
            agent_class_obj = PPOActorTransformer
        else:
            raise ValueError(f"Invalid agent class: {agent_class}")
        
        self.actor = agent_class_obj(action_dim, hidden_dim, **agent_kwargs).to(device)
        
        # FIX: Use _process_parameters instead of direct parameter loading
        self.agent_params = self._process_parameters(agent_params)
        self.verify_loaded_parameters()
        self.verify_architecture()  # Add this line

        # No jitting in PyTorch - we'll use the model directly
        self.actor_apply = self.actor

        # Test the model
        #self.test_model_with_known_input()


        self.reset()

    def verify_loaded_parameters(self):
        """Verify that loaded parameters match the model architecture"""
        print("=== PARAMETER VERIFICATION ===")
        
        model_state_dict = self.actor.state_dict()
        loaded_keys = set(self.agent_params.keys())
        model_keys = set(model_state_dict.keys())
        
        print(f"Loaded parameters: {len(loaded_keys)}")
        print(f"Model parameters: {len(model_keys)}")
        
        # Check matching parameters
        matched = loaded_keys.intersection(model_keys)
        print(f"Matched parameters: {len(matched)}")
        
        # Check missing parameters (in model but not loaded)
        missing = model_keys - loaded_keys
        print(f"Missing parameters: {len(missing)}")
        if missing:
            print("Missing parameter keys:")
            for key in sorted(missing)[:10]:  # Show first 10
                print(f"  {key}: {model_state_dict[key].shape}")
        
        # Check extra parameters (loaded but not in model)
        extra = loaded_keys - model_keys
        print(f"Extra parameters: {len(extra)}")
        if extra:
            print("Extra parameter keys:")
            for key in sorted(extra)[:10]:  # Show first 10
                print(f"  {key}: {self.agent_params[key].shape}")
        
        # Check parameter values
        print("\nParameter value ranges:")
        for key in sorted(matched):
            if 'weight' in key or 'bias' in key:
                loaded_val = self.agent_params[key]
                model_val = model_state_dict[key]
                print(f"  {key}: loaded [{loaded_val.min():.6f}, {loaded_val.max():.6f}], "
                    f"model [{model_val.min():.6f}, {model_val.max():.6f}]")
                # Check if values are actually different
                if torch.allclose(loaded_val, model_val, atol=1e-6):
                    print(f"    ✓ Values match")
                else:
                    print(f"    ✗ VALUES DIFFER!")
        
        print("==============================")

    def _process_parameters(self, agent_params):
        """Handle different parameter file structures"""
        if agent_params is None:
            print("No parameters provided, using random initialization")
            return {}
        
        print(f"Processing parameters: {len(agent_params)} keys")
        
        # Case 1: Flax-style flattened keys with commas
        if any(',' in key for key in agent_params.keys()):
            print("Detected Flax-style flattened parameters")
            torch_state_dict = self._convert_flax_to_torch_params(agent_params)
            
            # Debug: print first few converted keys
            print("First 5 converted parameter keys:")
            for i, key in enumerate(list(torch_state_dict.keys())[:5]):
                print(f"  {key}: {torch_state_dict[key].shape}")
            
            # Try to load the converted state dict
            try:
                missing_keys, unexpected_keys = self.actor.load_state_dict(torch_state_dict, strict=False)
                print(f"Parameter loading completed")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                if missing_keys:
                    print("Missing keys:", missing_keys)
                if unexpected_keys:
                    print("Unexpected keys:", unexpected_keys[:10])  # First 10 only

                self._initialize_missing_parameters()
                
                return torch_state_dict
            except Exception as e:
                print(f"Error loading parameters: {e}")
                print("Continuing with random initialization...")
                return {}
        
        # Case 2: Direct PyTorch state dict
        if any(key.startswith('input_layers') or key.startswith('rnn') for key in agent_params.keys()):
            print("Loading PyTorch state dict directly")
            self.actor.load_state_dict(agent_params)
            return agent_params
        
        # Case 3: Nested dictionary structure
        print("Treating as nested parameter structure")

        # Initialize missing parameters after loading

        return self._load_actor_params(agent_params)

    def _convert_flax_to_torch_params(self, flax_params):
        """Properly convert Flax-style parameter names to PyTorch format"""
        torch_state_dict = {}
        
        print("=== PARAMETER CONVERSION DEBUG ===")
        print(f"Total Flax parameters: {len(flax_params)}")
        
        # Print all Flax parameter keys for inspection
        print("Flax parameter keys:")
        for i, key in enumerate(sorted(flax_params.keys())):
            if i < 50:  # Show first 20 keys
                print(f"  {key}: {flax_params[key].shape}")
            elif i == 50:
                print("  ... (showing first 50 only)")
        
        # Key mappings from Flax to PyTorch
        key_mappings = {
            # Embedder
            'Embedder_0,Dense_0,kernel': 'embedder.0.weight',
            'Embedder_0,Dense_0,bias': 'embedder.0.bias', 
            'Embedder_0,LayerNorm_0,scale': 'embedder.1.weight',
            'Embedder_0,LayerNorm_0,bias': 'embedder.1.bias',
            
            # Transformer layer 0
            'ScannedTransformer_0,encoders_0,self_attn,query,kernel': 'transformer.encoders.0.self_attn.q_linear.weight',
            'ScannedTransformer_0,encoders_0,self_attn,key,kernel': 'transformer.encoders.0.self_attn.k_linear.weight',
            'ScannedTransformer_0,encoders_0,self_attn,value,kernel': 'transformer.encoders.0.self_attn.v_linear.weight',
            'ScannedTransformer_0,encoders_0,self_attn,out,kernel': 'transformer.encoders.0.self_attn.out_linear.weight',
            'ScannedTransformer_0,encoders_0,self_attn,out,bias': 'transformer.encoders.0.self_attn.out_linear.bias',
            
            'ScannedTransformer_0,encoders_0,linear_0,kernel': 'transformer.encoders.0.linear1.weight',
            'ScannedTransformer_0,encoders_0,linear_0,bias': 'transformer.encoders.0.linear1.bias',
            'ScannedTransformer_0,encoders_0,linear_1,kernel': 'transformer.encoders.0.linear2.weight', 
            'ScannedTransformer_0,encoders_0,linear_1,bias': 'transformer.encoders.0.linear2.bias',
            
            'ScannedTransformer_0,encoders_0,norm1,scale': 'transformer.encoders.0.norm1.weight',
            'ScannedTransformer_0,encoders_0,norm1,bias': 'transformer.encoders.0.norm1.bias',
            'ScannedTransformer_0,encoders_0,norm2,scale': 'transformer.encoders.0.norm2.weight',
            'ScannedTransformer_0,encoders_0,norm2,bias': 'transformer.encoders.0.norm2.bias',

            # Transformer layer 1
            'ScannedTransformer_0,encoders_1,self_attn,query,kernel': 'transformer.encoders.1.self_attn.q_linear.weight',
            'ScannedTransformer_0,encoders_1,self_attn,key,kernel': 'transformer.encoders.1.self_attn.k_linear.weight',
            'ScannedTransformer_0,encoders_1,self_attn,value,kernel': 'transformer.encoders.1.self_attn.v_linear.weight',
            'ScannedTransformer_0,encoders_1,self_attn,out,kernel': 'transformer.encoders.1.self_attn.out_linear.weight',
            'ScannedTransformer_0,encoders_1,self_attn,out,bias': 'transformer.encoders.1.self_attn.out_linear.bias',
            'ScannedTransformer_0,encoders_1,linear_0,kernel': 'transformer.encoders.1.linear1.weight',
            'ScannedTransformer_0,encoders_1,linear_0,bias': 'transformer.encoders.1.linear1.bias',
            'ScannedTransformer_0,encoders_1,linear_1,kernel': 'transformer.encoders.1.linear2.weight',
            'ScannedTransformer_0,encoders_1,linear_1,bias': 'transformer.encoders.1.linear2.bias',
            'ScannedTransformer_0,encoders_1,norm1,scale': 'transformer.encoders.1.norm1.weight',
            'ScannedTransformer_0,encoders_1,norm1,bias': 'transformer.encoders.1.norm1.bias',
            'ScannedTransformer_0,encoders_1,norm2,scale': 'transformer.encoders.1.norm2.weight',
            'ScannedTransformer_0,encoders_1,norm2,bias': 'transformer.encoders.1.norm2.bias',
            
            # Output layer
            'Dense_0,kernel': 'output_layer.weight',
            'Dense_0,bias': 'output_layer.bias',
        }
        
        converted_count = 0
        for flax_key, value in flax_params.items():
            torch_key = None
            
            # Remove 'params,' prefix if present
            clean_key = flax_key[7:] if flax_key.startswith('params,') else flax_key
            
            # Try exact mapping first
            for flax_pattern, torch_pattern in key_mappings.items():
                if flax_pattern in clean_key:
                    torch_key = torch_pattern
                    break
            
            # If no exact mapping found, use generic conversion
            if torch_key is None:
                torch_key = clean_key
                torch_key = torch_key.replace('kernel', 'weight')
                torch_key = torch_key.replace('scale', 'weight')
                torch_key = torch_key.replace('bias', 'bias')
                torch_key = torch_key.replace(',', '.')
            
            # Convert value to tensor
            if hasattr(value, '__array__'):
                tensor_value = torch.from_numpy(np.array(value))
            else:
                tensor_value = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
            
            # Handle weight transposition for linear layers
            if 'weight' in torch_key and tensor_value.dim() == 2:
                # Don't transpose attention output weights - they're already correct
                if not ('self_attn.out_linear.weight' in torch_key):
                    tensor_value = tensor_value.T
            
            # Handle attention weight reshaping
            if 'weight' in torch_key and 'self_attn' in torch_key:
                if tensor_value.dim() == 3:
                    if 'q_linear' in torch_key or 'k_linear' in torch_key or 'v_linear' in torch_key:
                        # Input projection: [hidden, heads, head_dim] -> [hidden, heads * head_dim]
                        hidden_dim, num_heads, head_dim = tensor_value.shape
                        tensor_value = tensor_value.reshape(hidden_dim, num_heads * head_dim)
                    elif 'out_linear' in torch_key:
                        # Output projection: [heads, head_dim, hidden] -> [heads * head_dim, hidden]
                        num_heads, head_dim, hidden_dim = tensor_value.shape
                        tensor_value = tensor_value.reshape(num_heads * head_dim, hidden_dim)
            
            torch_state_dict[torch_key] = tensor_value
            converted_count += 1
            
            if converted_count <= 50:  # Show first 10 conversions
                print(f"Converted: {flax_key} -> {torch_key} ({tensor_value.shape})")
        
        print(f"Total converted parameters: {converted_count}")
        print("=====================================")
        
        return torch_state_dict

    def _initialize_missing_parameters(self):
        """Initialize missing parameters to match JAX defaults"""
        print("Initializing missing parameters...")

        import torch.nn as nn
        
        # The missing attention output biases should be initialized to zero
        missing_params = [
            'transformer.encoders.0.self_attn.out_linear.bias',
            'transformer.encoders.1.self_attn.out_linear.bias'
        ]
        
        for param_name in missing_params:
            try:
                # Navigate to the parameter
                modules = param_name.split('.')
                current_module = self.actor
                
                for module_name in modules[:-1]:
                    if hasattr(current_module, module_name):
                        current_module = getattr(current_module, module_name)
                    else:
                        # Try integer indexing for lists
                        try:
                            current_module = current_module[int(module_name)]
                        except (ValueError, IndexError, KeyError):
                            print(f"Could not find module: {module_name}")
                            break
                
                # Initialize the bias parameter
                if hasattr(current_module, modules[-1]):
                    param = getattr(current_module, modules[-1])
                    if param is not None:
                        nn.init.constant_(param, 0.0)
                        print(f"✓ Initialized {param_name} to zero")
                else:
                    print(f"✗ Could not initialize {param_name}")
                    
            except Exception as e:
                print(f"Error initializing {param_name}: {e}")

    def _load_actor_params(self, params):
        """Load actor parameters for nested structure"""
        try:
            # Try to load as PyTorch state dict
            self.actor.load_state_dict(params)
            print("Successfully loaded PyTorch state dict")
            return params
        except (RuntimeError, KeyError) as e:
            print(f"Failed to load as PyTorch state dict: {e}")
            return {}

    def debug_model_outputs(self, obs, avail_actions, dones):
        """Debug the model's internal computations - simplified version"""
        print("=== DEBUG MODEL OUTPUTS ===")
        
        with torch.no_grad():
            print(f"Input obs shape: {obs.shape}")
            print(f"Hidden state shape: {self.hidden.shape}")
            
            # Just run the full forward pass without intermediate debugging
            new_hidden, logits = self.actor(self.hidden, (obs, dones, avail_actions))
            
            print(f"Output logits shape: {logits.shape}")
            print(f"Logits: {logits.squeeze().tolist()}")
            
            # Action masking
            unavail_actions = 1 - avail_actions.float()
            action_logits = logits - (unavail_actions * 1e10)
            
            action = torch.argmax(action_logits, dim=-1)
            print(f"Selected action: {action.item()}")
        
        print("==============================")

    def debug_observation_structure(self, obs_tensor):
            """Debug the observation tensor structure after batching"""
            print("=== OBSERVATION DEBUG ===")
            print(f"Observation tensor shape: {obs_tensor.shape}")
            print(f"Observation tensor dtype: {obs_tensor.dtype}")
            print(f"Observation range: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")
            
            if self.matrix_obs:
                # For matrix observations: [batch_size, num_actors, num_entities, obs_size]
                batch_size, num_actors, num_entities, obs_size = obs_tensor.shape
                print(f"Matrix observation: batch={batch_size}, actors={num_actors}, entities={num_entities}, obs_size={obs_size}")
                
                # Show first actor's first few entities
                print("First actor, first 3 entities:")
                for i in range(min(3, num_entities)):
                    entity_data = obs_tensor[0, 0, i].cpu().numpy()
                    print(f"  Entity {i}: {entity_data}")
            else:
                # For flattened observations: [batch_size, num_actors, feature_dim]
                batch_size, num_actors, feature_dim = obs_tensor.shape
                print(f"Flattened observation: batch={batch_size}, actors={num_actors}, feature_dim={feature_dim}")
                
                # Show first actor's first few features
                print("First actor, first 10 features:")
                actor_data = obs_tensor[0, 0, :10].cpu().numpy()
                print(f"  {actor_data}")
            
            print("==========================")

    def verify_architecture(self):
        """Verify that the PyTorch model matches the expected architecture"""
        print("=== MODEL ARCHITECTURE VERIFICATION ===")
        
        # Check key components
        components = [
            ('embedder.0.weight', (64, 6)),
            ('embedder.0.bias', (64,)),
            ('embedder.1.weight', (64,)),
            ('embedder.1.bias', (64,)),
            ('transformer.encoders.0.self_attn.q_linear.weight', (64, 64)),
            ('transformer.encoders.0.self_attn.k_linear.weight', (64, 64)),
            ('transformer.encoders.0.self_attn.v_linear.weight', (64, 64)),
            ('transformer.encoders.0.self_attn.out_linear.weight', (64, 64)),
            ('transformer.encoders.0.self_attn.out_linear.bias', (64,)),
            ('output_layer.weight', (5, 64)),
            ('output_layer.bias', (5,)),
        ]
        
        model_state = self.actor.state_dict()
        
        for param_name, expected_shape in components:
            if param_name in model_state:
                actual_shape = tuple(model_state[param_name].shape)
                if actual_shape == expected_shape:
                    print(f"✓ {param_name}: {actual_shape}")
                else:
                    print(f"✗ {param_name}: {actual_shape} (expected {expected_shape})")
            else:
                print(f"✗ {param_name}: NOT FOUND")
        
        print("=====================================")

    def debug_raw_observation(self, obs_dict):
        """Debug the raw observation dictionary before preprocessing"""
        print("=== RAW OBSERVATION DEBUG ===")
        
        # Check if we actually have a dictionary
        if not isinstance(obs_dict, dict):
            print(f"ERROR: Expected dictionary but got {type(obs_dict)}: {obs_dict}")
            print("==========================")
            return
        
        for agent_name, agent_obs in obs_dict.items():
            print(f"Agent: {agent_name}")
            
            # Check if agent_obs is a dictionary
            if not isinstance(agent_obs, dict):
                print(f"  ERROR: Agent observation is not a dictionary: {type(agent_obs)}: {agent_obs}")
                continue
                
            print(f"  Observation keys: {list(agent_obs.keys())}")
            
            # Check basic fields
            basic_fields = ['x', 'y', 'z', 'rph_z']
            for field in basic_fields:
                if field in agent_obs:
                    print(f"  {field}: {agent_obs[field]:.3f}")
                else:
                    print(f"  {field}: MISSING")
            
            # Check agent observations (other agents)
            agent_keys = [k for k in agent_obs.keys() if k.startswith('agent_') and 'dx' in k]
            print(f"  Other agents observed: {len(agent_keys)}")
            for agent_key in agent_keys[:2]:  # Show first 2
                print(f"    {agent_key}: {agent_obs[agent_key]:.3f}")
            
            # Check landmark observations
            landmark_keys = [k for k in agent_obs.keys() if k.startswith('landmark_')]
            landmark_tracking = [k for k in landmark_keys if 'tracking' in k]
            landmark_ranges = [k for k in landmark_keys if 'range' in k]
            
            print(f"  Landmark tracking data: {len(landmark_tracking)}")
            print(f"  Landmark range data: {len(landmark_ranges)}")
            
            # Show landmark data
            for i, landmark_key in enumerate(landmark_tracking[:2]):  # Show first 2
                print(f"    {landmark_key}: {agent_obs[landmark_key]:.3f}")
            
            for i, range_key in enumerate(landmark_ranges[:2]):  # Show first 2
                print(f"    {range_key}: {agent_obs[range_key]:.3f}")
        
        print("=============================")

    def debug_matrix_observation(self, obs_tensor):
        """Debug matrix observation structure"""
        print("=== MATRIX OBSERVATION DEBUG ===")
        print(f"Matrix obs shape: {obs_tensor.shape}")
        print(f"Matrix obs range: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")
        
        # Check entity ordering
        num_agents = len(self.agent_list)
        num_landmarks = len(self.landmark_list)
        
        print(f"Expected: {num_agents} agents + {num_landmarks} landmarks = {num_agents + num_landmarks} entities")
        print(f"Actual: {obs_tensor.shape[-2]} entities")
        
        if obs_tensor.shape[-2] != num_agents + num_landmarks:
            print("⚠️  Entity count mismatch!")
        print("================================")

    def debug_tensor_shapes(self, obs_batched, avail_actions, dones):
        """Debug all tensor shapes before forward pass"""
        print("=== TENSOR SHAPES DEBUG ===")
        print(f"obs_batched: {obs_batched.shape}")
        print(f"hidden: {self.hidden.shape}")
        print(f"avail_actions: {avail_actions.shape}")
        print(f"dones: {dones.shape}")
        
        # Check if shapes are compatible
        if self.matrix_obs:
            # For transformer: [batch, seq, entities, features]
            expected_entities = len(self.agent_list) + len(self.landmark_list)
            if obs_batched.shape[-2] != expected_entities:
                print(f"⚠️ Entity count mismatch: expected {expected_entities}, got {obs_batched.shape[-2]}")
        print("===========================")

    def test_model_with_known_input(self):
        """Test the model with a known input to see if it produces reasonable outputs"""
        print("=== MODEL TEST WITH KNOWN INPUT ===")
        
        # Create a simple test observation
        test_obs = {
            'agent_0': {
                'x': 10.0, 'y': 10.0, 'z': -5.0, 'rph_z': 0.0,
                'agent_1_dx': 20.0, 'agent_1_dy': 0.0, 'agent_1_dz': 0.0,
                'landmark_0_tracking_x': 50.0, 'landmark_0_tracking_y': 50.0, 
                'landmark_0_tracking_z': 10.0, 'landmark_0_range': 60.0,
            }
        }
        
        # Reset the model
        self.reset()
        
        # Get action
        with torch.no_grad():
            action = self.step(test_obs, done=False)
        
        print(f"Test action: {action}")
        
        # Check if action distribution makes sense
        print("Running multiple steps to check consistency...")
        actions = []
        for i in range(10):
            action = self.step(test_obs, done=False)
            actions.append(action['agent_0'])
            print(f"Step {i}: action = {action['agent_0']}")
        
        action_counts = {}
        for act in actions:
            action_counts[act] = action_counts.get(act, 0) + 1
        
        print(f"Action distribution: {action_counts}")
        print("=====================================")

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == 'cuda':
                torch.cuda.manual_seed(seed)
        
        # Initialize hidden state - handle different architectures
        if hasattr(self.actor, 'initialize_hidden'):
            # Check if initialize_hidden requires seq_len argument
            import inspect
            sig = inspect.signature(self.actor.initialize_hidden)
            if len(sig.parameters) == 2:  # Requires batch_size and seq_len (transformer)
                self.hidden = self.actor.initialize_hidden(len(self.actors_list), 1).to(self.device)
            else:  # Only requires batch_size (RNN)
                self.hidden = self.actor.initialize_hidden(len(self.actors_list)).to(self.device)
        else:
            # Default hidden state initialization
            if hasattr(self.actor, 'hidden_size'):
                hidden_size = self.actor.hidden_size
            elif hasattr(self.actor, 'hidden_dim'):
                hidden_size = self.actor.hidden_dim
            else:
                hidden_size = self.hidden_dim
                
            # For transformer, we need [batch_size, seq_len, hidden_dim]
            if hasattr(self.actor, 'transformer'):
                self.hidden = torch.zeros(len(self.actors_list), 1, hidden_size, device=self.device)
            else:
                # For RNN, we need [batch_size, hidden_dim]
                self.hidden = torch.zeros(len(self.actors_list), hidden_size, device=self.device)

    def step(self, obs: Dict, done: bool, avail_actions: Optional[Dict] = None) -> Dict[str, int]:
        # Debug raw observation FIRST (before any processing)
        if isinstance(obs, dict) and 'agent_0' in obs:
            if isinstance(obs['agent_0'], dict):
                self.debug_raw_observation(obs)
            else:
                print(f"WARNING: obs['agent_0'] is not a dictionary: {type(obs['agent_0'])}")

        # Handle available actions
        if avail_actions is None:
            avail_actions = torch.ones((len(self.actors_list), self.action_dim), dtype=torch.float32, device=self.device)
        else:
            avail_actions = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in avail_actions.items()}
            avail_actions = batchify(avail_actions, self.actors_list, len(self.actors_list))

        # Convert done to tensor
        dones = torch.full((len(self.actors_list),), done, dtype=torch.bool, device=self.device)

        # Preprocess observations
        obs_processed = {}
        for agent, o in obs.items():
            if isinstance(o, dict):
                obs_processed[agent] = self.preprocess_obs(agent, o)
            else:
                print(f"ERROR: Observation for {agent} is not a dictionary: {type(o)}")
                # Create a default observation to avoid crash
                obs_processed[agent] = self.preprocess_obs(agent, {})

        # Batch observations
        if self.matrix_obs:
            obs_batched = batchify_transformer(obs_processed, self.actors_list, len(self.actors_list))
            # Add sequence dimension: [1, num_actors, num_entities, obs_size]
            obs_batched = obs_batched.unsqueeze(0).to(self.device)
        else:
            obs_batched = batchify(obs_processed, self.actors_list, len(self.actors_list))
            # Add sequence dimension: [1, num_actors, feature_dim]
            obs_batched = obs_batched.unsqueeze(0).to(self.device)

        # Handle dones
        dones = dones.unsqueeze(0).to(self.device)  # [1, num_actors]
        
        # Ensure hidden state is on correct device
        self.hidden = self.hidden.to(self.device)

        # Debug batched observation (now a tensor)
        self.debug_observation_structure(obs_batched)
        
        # Debug model internals
        self.debug_model_outputs(obs_batched, avail_actions, dones)

        # Add this before the forward pass
        self.debug_tensor_shapes(obs_batched, avail_actions, dones)

        # Forward pass through actor
        with torch.no_grad():
            self.hidden, logits = self.actor_apply(self.hidden, (obs_batched, dones, avail_actions))

        # Get actions (argmax)
        print(f"Step - final logits shape: {logits.shape}")
        action = torch.argmax(logits, dim=-1)
        print(f"Step - final action: {action.item()}")

        # Unbatch actions
        action = unbatchify(action.cpu(), self.actors_list, self.num_envs, len(self.actors_list))
        action = {k: int(v.squeeze()) for k, v in action.items()}

        return action

    def preprocess_obs(self, agent_name: str, obs: Dict) -> torch.Tensor:
        # Pre-calculate sizes
        obs_size = 6
        total_obs_size = (len(self.agent_list) + len(self.landmark_list)) * obs_size

        if self.add_agent_id:
            total_obs_size += len(self.agent_list)

        # Preallocate observation array
        obs_array = torch.zeros(total_obs_size, device=self.device)

        # Index for filling the array
        idx = 0

        # Add other agents' observations
        for agent in self.agent_list:
            if agent == agent_name:
                # Add self observation
                obs_array[idx:idx + obs_size] = torch.tensor([
                    obs["x"] * self.pos_norm,
                    obs["y"] * self.pos_norm,
                    obs["z"] * self.pos_norm,
                    obs["rph_z"] * self.pos_norm,
                    1,  # is agent
                    1   # is self
                ], device=self.device)
                idx += obs_size
            else:
                obs_array[idx:idx + obs_size] = torch.tensor([
                    obs[f"{agent}_dx"] * self.pos_norm,
                    obs[f"{agent}_dy"] * self.pos_norm,
                    obs[f"{agent}_dz"] * self.pos_norm,
                    math.sqrt(
                        obs[f"{agent}_dx"] ** 2 +
                        obs[f"{agent}_dy"] ** 2 +
                        obs[f"{agent}_dz"] ** 2
                    ) * self.pos_norm * self.ranges_mask,
                    1,  # is agent
                    0   # is self
                ], device=self.device)
                idx += obs_size

        # Add landmarks' observations
        for landmark in self.landmark_list:
            obs_array[idx:idx + obs_size] = torch.tensor([
                (obs["x"] - obs[f"{landmark}_tracking_x"]) * self.pos_norm,
                (obs["y"] - obs[f"{landmark}_tracking_y"]) * self.pos_norm,
                (obs["z"] - obs[f"{landmark}_tracking_z"]) * self.pos_norm,
                obs[f"{landmark}_range"] * self.pos_norm * self.ranges_mask,
                0,  # is agent
                0   # is self
            ], device=self.device)
            idx += obs_size

        # Add agent id
        if self.add_agent_id:
            agent_id = torch.tensor(
                [1 if agent == agent_name else 0 for agent in self.agent_list],
                device=self.device
            )
            obs_array[idx:idx + len(self.agent_list)] = agent_id

        if self.matrix_obs:
            obs_array = obs_array.reshape((len(self.agent_list) + len(self.landmark_list), obs_size))

        # Check for NaN values
        if torch.isnan(obs_array).any():
            raise ValueError(f"NaN in the observation of agent {agent_name}")

        return obs_array

    def save_model(self, filename: Union[str, os.PathLike]):
        """Save the model state dict"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'hidden_dim': self.hidden_dim,
            'action_dim': self.action_dim,
            'agent_list': self.agent_list,
            'landmark_list': self.landmark_list,
            'actors_list': self.actors_list,
        }, filename)

    def load_model(self, filename: Union[str, os.PathLike]):
        """Load the model state dict"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])