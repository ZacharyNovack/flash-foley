import torch
from safetensors.torch import load_file

from torch.nn.utils import remove_weight_norm

def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    found_keys =  {}
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
            found_keys[key] = model_state_dict[key].numel()
        elif key in model_state_dict and "project_in" in key:
            # this to cover the edge case where we are trying to load in a model
            # with a different project_in embedding to the DiT than the checkpoint
            # while this is normally incorrect behavior, in the case of loading in a model
            # with more input concat parameters than the checkpoint, we can still partially
            # initialize this linear layer by copying over the first weights
            if state_dict[key].shape[1] <= model_state_dict[key].shape[1] and state_dict[key].shape[0] == model_state_dict[key].shape[0]:
                print(f"Partially loading weights for {key}, checkpoint shape: {state_dict[key].shape}, model shape: {model_state_dict[key].shape}")
                model_state_dict[key][:, :state_dict[key].shape[1]] = state_dict[key]
                found_keys[key] = state_dict[key].shape[0] * state_dict[key].shape[1]
            else:
                print(f"Skipping weights for {key}, checkpoint shape: {state_dict[key].shape}, model shape: {model_state_dict[key].shape}")
    if len(found_keys) == 0:
        missing_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        if missing_keys:
            raise ValueError(f"None of the keys in the state_dict match the model's state_dict. Missing keys: {missing_keys}")
    elif len(found_keys) < len(state_dict):
        missing_keys = set(state_dict.keys()) - set(found_keys.keys())
        if missing_keys:
            print(f"Some keys in the state_dict did not match the model's state_dict. Missing keys: {missing_keys}")

    # now print keys that are in the model but not in the state_dict
    extra_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    if extra_keys:
        print(f"Extra keys in the model's state_dict that are not in the state_dict: {extra_keys}")
    
    model.load_state_dict(model_state_dict, strict=True)

def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    
    return state_dict

def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)

    return model

try:
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True
except Exception as e:
    pass

# Get torch.compile flag from environment variable ENABLE_TORCH_COMPILE

import os
enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"

def compile(function, *args, **kwargs):
    
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function

# Sampling functions copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py under MIT license
# License can be found in LICENSES/LICENSE_META.txt

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def next_power_of_two(n):
    return 2 ** (n - 1).bit_length()

def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64