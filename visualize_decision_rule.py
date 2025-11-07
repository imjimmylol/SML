import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import os
from typing import Dict, Tuple

def prepare_decision_rule_grid(
    state_dict_base: Dict,
    variable_to_vary: str,
    agent_idx: int = 0,
    n_grid_points: int = 100,
    n_ability_levels: int = 7,
    grid_min: float = None,
    grid_max: float = None
):
    """
    Prepare grid for decision rule visualization.
    
    Parameters:
    -----------
    state_dict_base : dict
        Base state dictionary to use as template
    variable_to_vary : str
        Which variable to vary ('moneydisposable', 'savings', or 'ability')
    agent_idx : int
        Which agent to vary (default: 0)
    n_grid_points : int
        Number of grid points for the varying variable
    n_ability_levels : int
        Number of different ability levels to show
    grid_min, grid_max : float
        Min/max values for the grid (if None, uses data range)
    """
    
    # Extract base tensors and convert to numpy for manipulation
    moneydisposable = state_dict_base["moneydisposable"].cpu().numpy()
    savings = state_dict_base["savings"].cpu().numpy()
    ability = state_dict_base["ability"].cpu().numpy()
    
    # Get batch size and number of agents
    if len(moneydisposable.shape) == 1:
        moneydisposable = moneydisposable.reshape(1, -1)
        savings = savings.reshape(1, -1)
        ability = ability.reshape(1, -1)
    
    batch_size, n_agents = moneydisposable.shape
    
    # Use first batch as template
    m_base = moneydisposable[0]
    s_base = savings[0]
    a_base = ability[0]
    
    # Expand dimensions for grid
    # Shape will be (n_grid_points, n_ability_levels, n_agents)
    m_grid = np.repeat(np.expand_dims(np.expand_dims(m_base, 0), 0), n_grid_points, axis=0)
    m_grid = np.repeat(m_grid, n_ability_levels, axis=1)
    
    s_grid = np.repeat(np.expand_dims(np.expand_dims(s_base, 0), 0), n_grid_points, axis=0)
    s_grid = np.repeat(s_grid, n_ability_levels, axis=1)
    
    a_grid = np.repeat(np.expand_dims(a_base, 0), n_ability_levels, axis=0)
    a_grid = np.repeat(np.expand_dims(a_grid, 0), n_grid_points, axis=0)
    
    # Set up the varying dimension
    if variable_to_vary == 'moneydisposable':
        if grid_min is None:
            grid_min = m_base.min() * 0.1
        if grid_max is None:
            grid_max = m_base.max() * 3.0
        # Vary money for agent_idx across grid points
        m_grid[:, :, agent_idx] = np.expand_dims(
            np.linspace(grid_min, grid_max, n_grid_points), 1
        )
        # Vary ability for agent_idx across ability levels
        a_grid[:, :, agent_idx] = np.linspace(
            a_base.min() * 0.5, 
            a_base.max() * 2.0, 
            n_ability_levels
        )
        
    elif variable_to_vary == 'savings':
        if grid_min is None:
            grid_min = s_base.min() * 0.1
        if grid_max is None:
            grid_max = s_base.max() * 3.0
        # Vary savings for agent_idx
        s_grid[:, :, agent_idx] = np.expand_dims(
            np.linspace(grid_min, grid_max, n_grid_points), 1
        )
        # Vary ability for agent_idx across ability levels
        a_grid[:, :, agent_idx] = np.linspace(
            a_base.min() * 0.5,
            a_base.max() * 2.0,
            n_ability_levels
        )
        
    elif variable_to_vary == 'ability':
        # When varying ability, use money as the second dimension
        if grid_min is None:
            grid_min = a_base.min() * 0.5
        if grid_max is None:
            grid_max = a_base.max() * 2.0
        # Vary ability for agent_idx across grid points
        a_grid[:, :, agent_idx] = np.expand_dims(
            np.linspace(grid_min, grid_max, n_grid_points), 1
        )
        # Vary money for agent_idx across levels
        m_grid[:, :, agent_idx] = np.linspace(
            m_base.min() * 0.5,
            m_base.max() * 3.0,
            n_ability_levels
        )
    
    # Prepare other state variables
    tax_params = state_dict_base["tax_params"]
    if len(tax_params.shape) == 1:
        tax_params = tax_params.unsqueeze(0)
    tax_params_grid = tax_params[0:1].repeat(n_ability_levels, 1)
    
    is_superstar_vA = torch.zeros((n_ability_levels, n_agents), dtype=torch.bool)
    is_superstar_vB = torch.zeros((n_ability_levels, n_agents), dtype=torch.bool)
    
    ret = state_dict_base.get("ret", 0.06)
    if torch.is_tensor(ret):
        ret = ret.item()
    
    return {
        'moneydisposable_grid': m_grid,
        'savings_grid': s_grid,
        'ability_grid': a_grid,
        'tax_params_grid': tax_params_grid,
        'is_superstar_vA': is_superstar_vA,
        'is_superstar_vB': is_superstar_vB,
        'ret': ret,
        'n_grid_points': n_grid_points,
        'n_ability_levels': n_ability_levels,
        'agent_idx': agent_idx,
        'variable_to_vary': variable_to_vary
    }


def evaluate_decision_rules(
    model,
    grid_data: Dict,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate the model on the grid to get decision rules.
    """
    model.eval()
    
    n_grid_points = grid_data['n_grid_points']
    n_ability_levels = grid_data['n_ability_levels']
    agent_idx = grid_data['agent_idx']
    
    # Storage for results
    savings_decisions = np.zeros((n_grid_points, n_ability_levels))
    labor_decisions = np.zeros((n_grid_points, n_ability_levels))
    multiplier_decisions = np.zeros((n_grid_points, n_ability_levels))
    consumption_decisions = np.zeros((n_grid_points, n_ability_levels))
    
    with torch.no_grad():
        for i in range(n_grid_points):
            # Prepare state dict for this grid point
            state_dict = {
                'moneydisposable': torch.tensor(
                    grid_data['moneydisposable_grid'][i], 
                    dtype=torch.float32, 
                    device=device
                ),
                'savings': torch.tensor(
                    grid_data['savings_grid'][i], 
                    dtype=torch.float32,
                    device=device
                ),
                'ability': torch.tensor(
                    grid_data['ability_grid'][i],
                    dtype=torch.float32,
                    device=device
                ),
                'tax_params': grid_data['tax_params_grid'].to(device),
                'is_superstar_vA': grid_data['is_superstar_vA'].to(device),
                'is_superstar_vB': grid_data['is_superstar_vB'].to(device),
                'ret': grid_data['ret']
            }
            
            # Run model to get decisions
            from src.model import FiLMResNet2In
            import torch.nn.functional as F
            
            # Build inputs (you'll need to import your build_inputs function)
            features, conditions = build_inputs(
                state_dict['moneydisposable'],
                state_dict['ability'],
                state_dict['tax_params']
            )
            
            # Get model output
            out = model(features, conditions)
            
            # Apply activation functions
            savings_ratio = torch.sigmoid(out[..., 0])
            multiplier = F.softplus(out[..., 1]) + 1e-6
            labor = torch.sigmoid(out[..., 2])
            
            # Calculate consumption from money disposable and savings ratio
            consumption = state_dict['moneydisposable'] * (1 - savings_ratio)
            
            # Store results for the agent of interest
            savings_decisions[i, :] = savings_ratio[:, agent_idx].cpu().numpy()
            labor_decisions[i, :] = labor[:, agent_idx].cpu().numpy()
            multiplier_decisions[i, :] = multiplier[:, agent_idx].cpu().numpy()
            consumption_decisions[i, :] = consumption[:, agent_idx].cpu().numpy()
    
    return {
        'savings_ratio': savings_decisions,
        'labor': labor_decisions,
        'multiplier': multiplier_decisions,
        'consumption': consumption_decisions
    }


def plot_decision_rules(
    grid_data: Dict,
    results: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Create visualization of decision rules.
    """
    variable_to_vary = grid_data['variable_to_vary']
    n_grid_points = grid_data['n_grid_points']
    n_ability_levels = grid_data['n_ability_levels']
    agent_idx = grid_data['agent_idx']
    
    # Get x-axis values
    if variable_to_vary == 'moneydisposable':
        x_values = grid_data['moneydisposable_grid'][:, 0, agent_idx]
        x_label = 'Money Disposable'
        legend_label = 'Ability'
        legend_values = grid_data['ability_grid'][0, :, agent_idx]
    elif variable_to_vary == 'savings':
        x_values = grid_data['savings_grid'][:, 0, agent_idx]
        x_label = 'Savings'
        legend_label = 'Ability'
        legend_values = grid_data['ability_grid'][0, :, agent_idx]
    else:  # ability
        x_values = grid_data['ability_grid'][:, 0, agent_idx]
        x_label = 'Ability'
        legend_label = 'Money Disposable'
        legend_values = grid_data['moneydisposable_grid'][0, :, agent_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Decision Rules - Varying {x_label} for Agent {agent_idx}', fontsize=14)
    
    # Color map for different levels
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_ability_levels))
    
    # Plot 1: Savings Ratio
    ax = axes[0, 0]
    for j in range(n_ability_levels):
        ax.plot(x_values, results['savings_ratio'][:, j], 
                color=colors[j], linewidth=2,
                label=f'{legend_label}={legend_values[j]:.2f}')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Savings Ratio')
    ax.set_title('Savings Decision')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    
    # Plot 2: Labor Supply
    ax = axes[0, 1]
    for j in range(n_ability_levels):
        ax.plot(x_values, results['labor'][:, j],
                color=colors[j], linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Labor Supply')
    ax.set_title('Labor Decision')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Consumption
    ax = axes[1, 0]
    for j in range(n_ability_levels):
        ax.plot(x_values, results['consumption'][:, j],
                color=colors[j], linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Consumption')
    ax.set_title('Consumption Level')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Borrowing Constraint Multiplier
    ax = axes[1, 1]
    for j in range(n_ability_levels):
        ax.plot(x_values, results['multiplier'][:, j],
                color=colors[j], linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Lagrange Multiplier')
    ax.set_title('Borrowing Constraint Tightness')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Decision rules plot saved to {save_path}")
    
    plt.show()
    
    return fig


def build_inputs(moneydisposable, ability, tax_params, device='cuda', eps=1e-8):
    """
    Build inputs for the model (copied from your code).
    """
    # Ensure tensors
    moneydisposable = torch.as_tensor(moneydisposable, dtype=torch.float32, device=device)
    ability = torch.as_tensor(ability, dtype=torch.float32, device=device)
    tax_params = torch.as_tensor(tax_params, dtype=torch.float32, device=device)
    
    B, A = moneydisposable.shape
    
    # Min-Max normalization
    m_min = moneydisposable.min(dim=1, keepdim=True)[0]
    m_max = moneydisposable.max(dim=1, keepdim=True)[0]
    moneydisposable = (moneydisposable - m_min) / (m_max - m_min + eps)
    
    # Expand tax params
    condi = tax_params.unsqueeze(1).expand(-1, A, -1)
    
    # Aggregate info
    sum_info = torch.cat([moneydisposable, ability], dim=1)
    sum_info_rep = sum_info.unsqueeze(1).expand(-1, A, -1)
    
    # Individual info
    money_self = moneydisposable.unsqueeze(-1)
    ability_self = ability.unsqueeze(-1)
    
    features = torch.cat([sum_info_rep, money_self, ability_self], dim=2)
    
    return features, condi


# Main execution function
def visualize_model_decisions(
    model_path: str,
    state_path: str,
    model,
    output_dir: str = './figures',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Main function to create all decision rule visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print(f"✅ Model loaded from {model_path}")
    
    # Load state
    with open(state_path, "rb") as f:
        state_dict = pickle.load(f)
    print(f"✅ State loaded from {state_path}")
    
    # Create visualizations for different variables
    variables_to_plot = ['moneydisposable', 'savings', 'ability']
    
    for var in variables_to_plot:
        print(f"\n📊 Creating decision rules for varying {var}...")
        
        # Prepare grid
        grid_data = prepare_decision_rule_grid(
            state_dict_base=state_dict,
            variable_to_vary=var,
            agent_idx=0,  # Focus on first agent
            n_grid_points=100,
            n_ability_levels=5  # Reduced for clarity
        )
        
        # Evaluate model
        results = evaluate_decision_rules(model, grid_data, device)
        
        # Plot
        save_path = os.path.join(output_dir, f'decision_rules_{var}.png')
        plot_decision_rules(grid_data, results, save_path)
    
    print(f"\n✅ All visualizations complete! Check {output_dir}")


if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    from src.model import FiLMResNet2In
    
    # Initialize model
    AGENTS = 50
    state_dim = 2 * AGENTS + 2
    cond_dim = 5
    model = FiLMResNet2In(
        state_dim=state_dim,
        cond_dim=cond_dim,
        hidden_dim=128,
        num_res_blocks=3,
        output_dim=3,
        dropout=0.1
    )
    
    # Paths
    model_path = "./checkpoints/exp_transition/run/weights/model_final.pt"
    state_path = "./checkpoints/exp_transition/run/states/state_step_0.pkl"
    
    # Create visualizations
    visualize_model_decisions(model_path, state_path, model)