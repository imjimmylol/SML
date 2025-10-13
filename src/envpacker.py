import torch
from src.normalizer import RunningMeanStd
def packenv(money_disposable, v, tax_params, rms: RunningMeanStd):
    """
    Packs the environment state into a single tensor for model input.
    Args:
        money_disposable (torch.Tensor): shape (AGENTS, 1), disposable money.
        v (torch.Tensor): shape (AGENTS, 1), ability.
        tax_params (torch.Tensor): shape (5,) or (1, 5), tax parameters shared across all agents.
        rms (RunningMeanStd): Running mean and std for normalization (must be provided).
        # rms should be initialized w.r.t 2 feature sizes, in this case (money_disposable and v)
        # rms = RunningMeanStd(shape=(2,))
    Returns:
        torch.Tensor: Packed tensor of shape (AGENTS, 2*AGENTS + 2 + 5) containing:
                     - all agents' normalized money_disposable (AGENTS values)
                     - all agents' normalized v (AGENTS values)
                     - agent i's own normalized money_disposable (1 value)
                     - agent i's own normalized v (1 value)
                     - tax parameters (5 values)
    """
    # Combine and normalize
    combined = torch.cat([money_disposable, v], dim=1)  # (AGENTS, 2)
    rms.update(combined.cpu().numpy())  # Update rms with current batch
    normalized = rms.normalize(combined.cpu().numpy())  # (AGENTS, 2)
    normalized = torch.from_numpy(normalized).to(money_disposable.device)  # Back to tensor
    
    money_disposable_norm = normalized[:, 0]  # (AGENTS,)
    v_norm = normalized[:, 1]  # (AGENTS,)
    
    # Get number of agents
    AGENTS = money_disposable.shape[0]
    
    # Flatten tax_params to 1D if needed
    if tax_params.dim() > 1:
        tax_params = tax_params.flatten()  # (5,)
    
    # Ensure tax_params is on the same device
    tax_params = tax_params.to(money_disposable.device)
    
    # Vectorized construction:
    # Repeat global information for all agents
    money_global = money_disposable_norm.unsqueeze(0).expand(AGENTS, -1)  # (AGENTS, AGENTS)
    v_global = v_norm.unsqueeze(0).expand(AGENTS, -1)  # (AGENTS, AGENTS)
    
    # Individual information (each agent's own values)
    money_individual = money_disposable_norm.unsqueeze(1)  # (AGENTS, 1)
    v_individual = v_norm.unsqueeze(1)  # (AGENTS, 1)
    
    # Tax parameters repeated for all agents
    tax_params_repeated = tax_params.unsqueeze(0).expand(AGENTS, -1)  # (AGENTS, 5)
    
    # Concatenate all components
    packed_tensor = torch.cat([
        money_global,           # (AGENTS, AGENTS)
        v_global,               # (AGENTS, AGENTS)
        money_individual,       # (AGENTS, 1)
        v_individual,           # (AGENTS, 1)
        tax_params_repeated     # (AGENTS, 5)
    ], dim=1)  # (AGENTS, 2*AGENTS + 2 + 5)
    
    return packed_tensor, rms
    

def packenv_batch(money_disposable, v, tax_params, rms: RunningMeanStd):
    """
    Packs a batch of environment states into a single tensor for model input.
    
    Args:
        money_disposable (torch.Tensor): shape (BATCH_SIZE, AGENTS, 1), disposable money.
        v (torch.Tensor): shape (BATCH_SIZE, AGENTS, 1), ability.
        tax_params (torch.Tensor): shape (BATCH_SIZE, 5), tax params for each env in the batch.
        rms (RunningMeanStd): Running mean and std normalizer.

    Returns:
        torch.Tensor: Packed tensor of shape (BATCH_SIZE, AGENTS, 2*AGENTS + 2 + 5).
        RunningMeanStd: The updated normalizer.
    """
    # Get dimensions and device
    BATCH_SIZE, AGENTS, _ = money_disposable.shape
    device = money_disposable.device

    # 1. Combine and reshape for normalization
    # We flatten the batch and agent dimensions to update the normalizer
    combined = torch.cat([money_disposable, v], dim=2)  # Shape: (B, A, 2)
    combined_reshaped = combined.view(-1, 2)            # Shape: (B * A, 2)

    # 2. Update RMS and normalize the data
    rms.update(combined_reshaped.cpu().numpy())
    normalized_reshaped = rms.normalize(combined_reshaped.cpu().numpy())
    
    # 3. Reshape back to (B, A, 2) and extract features
    normalized = torch.from_numpy(normalized_reshaped).view(BATCH_SIZE, AGENTS, 2).to(device)
    money_disposable_norm = normalized[..., 0]  # Shape: (B, A)
    v_norm = normalized[..., 1]                 # Shape: (B, A)

    # 4. Prepare tensors for concatenation by expanding them to the correct shape
    # For each agent, we need a view of all other agents' states within its batch.
    money_global = money_disposable_norm.unsqueeze(1).expand(-1, AGENTS, -1) # Shape: (B, A, A)
    v_global = v_norm.unsqueeze(1).expand(-1, AGENTS, -1)                     # Shape: (B, A, A)
    
    # Each agent's own state.
    money_individual = money_disposable_norm.unsqueeze(2) # Shape: (B, A, 1)
    v_individual = v_norm.unsqueeze(2)                    # Shape: (B, A, 1)
    
    # Tax parameters, repeated for every agent within its respective environment.
    tax_params_repeated = tax_params.unsqueeze(1).expand(-1, AGENTS, -1) # Shape: (B, A, 5)

    # 5. Concatenate along the last dimension to form the final input tensor
    packed_tensor = torch.cat([
        money_global,           # (B, A, A)
        v_global,               # (B, A, A)
        money_individual,       # (B, A, 1)
        v_individual,           # (B, A, 1)
        tax_params_repeated     # (B, A, 5)
    ], dim=2)
    
    return packed_tensor, rms