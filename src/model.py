import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        # Generate gamma (scale) and beta (shift) from conditioning input
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net = nn.Linear(cond_dim, feature_dim)
    
    def forward(self, x, cond):
        """
        Args:
            x: features to be modulated, shape (batch, feature_dim)
            cond: conditioning variables, shape (batch, cond_dim)
        """
        gamma = self.gamma_net(cond)
        beta = self.beta_net(cond)
        return gamma * x + beta


class ResidualBlock(nn.Module):
    """Simple residual block with FiLM conditioning"""
    def __init__(self, hidden_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film1 = FiLMLayer(cond_dim, hidden_dim)
        self.film2 = FiLMLayer(cond_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x, cond):
        """
        Args:
            x: input features, shape (batch, hidden_dim)
            cond: conditioning variables (tax parameters), shape (batch, cond_dim)
        """
        residual = x
        
        # First layer with FiLM
        out = self.fc1(x)
        out = self.film1(out, cond)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second layer with FiLM
        out = self.fc2(out)
        out = self.film2(out, cond)
        
        # Residual connection
        out = out + residual
        out = self.activation(out)
        
        return out


class FiLMResNet(nn.Module):
    """Neural network with FiLM conditioning and residual blocks"""
    def __init__(self, AGENTS, hidden_dim=128, num_res_blocks=2, output_dim=1, dropout=0.1):
        super().__init__()
        
        # Calculate input dimensions
        self.AGENTS = AGENTS
        self.state_dim = 2 * AGENTS + 2  # 2*AGENTS + 2 state variables
        self.cond_dim = 5  # 5 exogenous tax parameters
        self.input_dim = self.state_dim + self.cond_dim
        
        # Initial projection for state variables
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # FiLM conditioning on the encoded state
        self.initial_film = FiLMLayer(self.cond_dim, hidden_dim)
        
        # Residual blocks with FiLM
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, self.cond_dim, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: input tensor, shape (batch, input_dim)
               where input_dim = 2*AGENTS + 2 + 5
        
        Returns:
            output: shape (batch, output_dim)
        """
        # Split input into state variables and conditioning variables
        state = x[:, :self.state_dim]  # (batch, 2*AGENTS + 2)
        cond = x[:, self.state_dim:]   # (batch, 5) - tax parameters
        
        # Encode state variables
        features = self.state_encoder(state)
        
        # Apply initial FiLM conditioning
        features = self.initial_film(features, cond)
        
        # Pass through residual blocks with FiLM
        for res_block in self.res_blocks:
            features = res_block(features, cond)
        
        # Generate output
        output = self.output_head(features)
        
        return output
    

    # # Configuration
    # AGENTS = 10
    # batch_size = 32
    # hidden_dim = 128
    # num_res_blocks = 3
    # output_dim = 1
    
    # # Create model
    # model = FiLMResNet(
    #     AGENTS=AGENTS,
    #     hidden_dim=hidden_dim,
    #     num_res_blocks=num_res_blocks,
    #     output_dim=output_dim,
    #     dropout=0.1
    # )