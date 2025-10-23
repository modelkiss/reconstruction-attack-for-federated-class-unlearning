"""
strategies.py

Aggregation strategies implementations:
- FedAvg: weighted average by num_samples
- FedProx: uses FedAvg aggregation but clients run with proximal term (mu)
- FedOpt: server-side optimizer (Adam) updates using aggregated client deltas treated as gradients
"""

from typing import List, Dict, Any
import copy
import torch
import torch.optim as optim


def fedavg_aggregate(global_state: Dict[str, torch.Tensor], client_results: List[Dict[str, Any]]):
    """
    FedAvg aggregation: weighted average of client model parameters by number of samples.
    client_results: list of dicts with keys: 'num_samples', 'state_dict'
    Returns new_state (state_dict)
    """
    total_samples = sum([r["num_samples"] for r in client_results])
    if total_samples == 0:
        return global_state

    new_state = {}
    # initialize accumulator as float tensors to allow weighted (float) accumulation
    for k, v in global_state.items():
        # place accumulator on same device as global_state[k], but use float dtype for accumulation
        new_state[k] = torch.zeros_like(v, dtype=torch.float32, device=v.device)

    # accumulate weighted params (convert client tensors to float)
    for r in client_results:
        weight = r["num_samples"] / total_samples
        client_state = r["state_dict"]
        for k in global_state.keys():
            # ensure client tensor is on same device as accumulator and cast to float for accumulation
            client_tensor = client_state[k].to(new_state[k].device).to(torch.float32)
            new_state[k] += client_tensor * weight

    # convert accumulators back to original dtype of global_state (for integer buffers preserve ints)
    final_state = {}
    for k, v in global_state.items():
        target_dtype = v.dtype
        tensor = new_state[k]
        # if original was integer type, round before casting to preserve integer semantics
        if target_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            # round to nearest integer safely
            tensor = torch.round(tensor).to(target_dtype)
        else:
            tensor = tensor.to(target_dtype)
        # ensure stored on CPU to be consistent with previous code that saved CPU tensors
        final_state[k] = tensor.cpu().clone()
    return final_state


class FedOptAggregator:
    """
    FedOpt aggregator: maintains a server optimizer that applies parameter updates using
    aggregated client updates treated as gradients (grad = -avg_delta).
    """

    def __init__(self, global_state: Dict[str, torch.Tensor], lr: float = 1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # create a "server model" containers (only param tensors needed)
        # store everything on CPU for server-side optimizer
        self.device = torch.device("cpu")
        # ensure param_list are float tensors and require_grad=True
        self.keys = list(global_state.keys())
        self.param_list = []
        for k in self.keys:
            p = global_state[k].detach().clone().to(self.device).to(torch.float32).requires_grad_(True)
            self.param_list.append(p)
        # use torch optimizer
        self.optimizer = optim.Adam(self.param_list, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, global_state: Dict[str, torch.Tensor], client_results: List[Dict[str, Any]]):
        """
        Apply one FedOpt aggregation step and return updated global_state.
        """
        total_samples = sum([r["num_samples"] for r in client_results]) or 1
        # compute avg_delta (on cpu)
        avg_delta = {}
        for k in global_state.keys():
            avg_delta[k] = torch.zeros_like(global_state[k], dtype=torch.float32)

        for r in client_results:
            weight = r["num_samples"] / total_samples
            client_state = r["state_dict"]
            # delta = client - global (both CPU); cast to float32 for arithmetic
            for k in global_state.keys():
                avg_delta[k] += (client_state[k].cpu().to(torch.float32) - global_state[k].cpu().to(torch.float32)) * weight

        # set gradient = -avg_delta and perform optimizer step on param_list
        for idx, k in enumerate(self.keys):
            if self.param_list[idx].grad is not None:
                self.param_list[idx].grad.zero_()
            # gradient direction: want to move global towards average client -> gradient = -(avg_delta)
            self.param_list[idx].grad = -avg_delta[k].to(self.device)

        self.optimizer.step()

        # build new global_state from param_list (cast back to original dtype/device)
        new_state = {}
        for idx, k in enumerate(self.keys):
            orig = global_state[k]
            updated = self.param_list[idx].detach().clone().to(orig.device)
            # cast back to original dtype (round if integer)
            if orig.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                updated = torch.round(updated).to(orig.dtype)
            else:
                updated = updated.to(orig.dtype)
            new_state[k] = updated
        return new_state

def fedprox_aggregate(global_state: Dict[str, torch.Tensor], client_results: List[Dict[str, Any]]):
    """
    FedProx uses the same aggregation as FedAvg (server-side), while clients add proximal term locally.
    Here we reuse FedAvg aggregation.
    """
    return fedavg_aggregate(global_state, client_results)




def dp_fedavg_aggregate(
    global_state: Dict[str, torch.Tensor],
    client_results: List[Dict[str, Any]],
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.0,
    generator: Optional[torch.Generator] = None,
):
    """
    Differentially-private FedAvg aggregation.

    Each client update is clipped to `clip_norm` and averaged. Gaussian noise with
    standard deviation `noise_multiplier * clip_norm` is added to each parameter.
    Returns a new global state dict on CPU.
    """

    if not client_results:
        return {k: v.detach().cpu().clone() for k, v in global_state.items()}

    device = torch.device("cpu")
    clip_norm = max(clip_norm, 1e-12)
    std = max(noise_multiplier, 0.0) * clip_norm
    generator = generator or torch.Generator(device=device)

    aggregated = {
        k: torch.zeros_like(v, dtype=torch.float32, device=device)
        for k, v in global_state.items()
    }

    for res in client_results:
        client_state = res["state_dict"]
        deltas = {}
        sq_norm = 0.0
        for key in global_state.keys():
            delta = client_state[key].cpu().to(torch.float32) - global_state[key].cpu().to(torch.float32)
            deltas[key] = delta
            sq_norm += float(torch.sum(delta ** 2).item())

        norm = sq_norm ** 0.5
        scale = 1.0
        if clip_norm > 0.0 and norm > clip_norm:
            scale = clip_norm / (norm + 1e-12)

        for key in aggregated.keys():
            aggregated[key] += deltas[key] * scale

    num_clients = max(1, len(client_results))
    new_state: Dict[str, torch.Tensor] = {}
    for key, base_tensor in global_state.items():
        update = aggregated[key] / float(num_clients)
        if std > 0.0:
            noise = torch.randn_like(update, generator=generator) * std
            update = update + noise
        updated = base_tensor.cpu().to(torch.float32) + update
        if base_tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            updated = torch.round(updated).to(base_tensor.dtype)
        else:
            updated = updated.to(base_tensor.dtype)
        new_state[key] = updated.cpu().clone()

    return new_state