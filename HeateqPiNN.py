import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# This section defines all hyperparameters and settings for the improved PINN model.
# We use a dataclass to store everything in one place for easy modification.
# The key improvements include better loss weighting, increased network capacity,
# and more training points to achieve better accuracy.
@dataclass
class Config:
    alpha: float = 1.0       # Thermal diffusivity constant in the heat equation
    n_hidden: int = 4        # Reduced number of hidden layers for better training
    width: int = 50          # Increased number of neurons per hidden layer
    N_f: int = 10000         # Increased collocation points for better PDE coverage
    N_b: int = 500           # Increased boundary points
    N_0: int = 2000          # Number of initial condition points
    epochs: int = 10000      # Increased training iterations for convergence
    lr: float = 1e-3         # Increased learning rate for faster training
    seed: int = 1234         # Random seed for reproducibility
    dtype = torch.float32    # Precision for computations
    device: str = "cpu"      # Device to run on (CPU or GPU)
    
    # Loss weights for balancing different components - these are crucial!
    # Higher weights for boundary and initial conditions ensure they are satisfied
    w_f: float = 1.0         # Weight for PDE residual loss
    w_b: float = 100.0       # Weight for boundary loss (much higher priority)
    w_0: float = 100.0       # Weight for initial condition loss (much higher priority)

cfg = Config()
torch.manual_seed(cfg.seed)  # Ensures reproducible results across runs

# This function calculates the exact solution of the 1D heat equation.
# It will be used later to compare the PINN's predictions against the true solution.
# The solution represents a sine wave that decays exponentially over time.
def exact_solution(x, t, alpha):
    """Calculate the exact analytical solution: u(x,t) = exp(-α*π²*t) * sin(π*x)"""
    return torch.exp(-alpha * (torch.pi**2) * t) * torch.sin(torch.pi * x)

# This class defines the neural network (PINN) that approximates the solution u(x,t).
# The network takes inputs x and t and outputs u
class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = 2      # Input dimensions: x and t
        out_dim = 1     # Output dimension: u(x,t)
        width = cfg.width
        n_hidden = cfg.n_hidden

        # Build the network: a list of linear layers with improved architecture
        layers = []
        layers.append(nn.Linear(in_dim, width))
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, out_dim))
        
        self.layers = nn.ModuleList(layers)
        self.act = nn.Tanh()  # Use tanh activation for smoothness (important for derivatives)
        
        # Xavier initialization for better training stability and convergence
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # Forward pass through the network - transforms input (x,t) to output u(x,t)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)  # No activation for the last layer
            else:
                x = self.act(layer(x))  # Apply tanh to hidden layers
        return x

    # Computes the PDE residual for the heat equation: f = u_t - alpha * u_xx
    def pde_residual(self, x, t, alpha=1.0):
        # Ensure gradients are computed correctly with proper tensor management
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        u = self.forward(torch.cat([x, t], dim=1))  # Get network prediction

        # Compute first derivatives using PyTorch autograd with improved settings
        u_t = torch.autograd.grad(
            outputs=u, inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # Compute second derivative u_xx by differentiating u_x
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]

        # Heat equation residual: should be zero when PDE is satisfied
        f = u_t - alpha * u_xx
        return f

# Generates N random points between 'low' and 'high'. Used for collocation, boundary, and initial points.
# This function creates the sampling points where we'll enforce our physics constraints.
def sample_uniform(N, low=0.0, high=1.0):
    return low + (high - low) * torch.rand(N, 1, dtype=cfg.dtype, device=cfg.device)

# Prepares the training datasets for the PINN with improved sampling strategy.
# We generate more points and distribute them more systematically for better coverage.
# Returns: interior points, left boundary, right boundary, and initial condition points.
def make_training_sets(cfg):
    # Interior points (collocation points) - where we enforce the PDE
    x_f = sample_uniform(cfg.N_f)  # Interior x values (0 to 1)
    t_f = sample_uniform(cfg.N_f)  # Interior t values (0 to 1)

    # Boundary points: more systematic sampling for better constraint enforcement
    # Left boundary (x=0) - temperature must be zero for all time
    t_b0 = sample_uniform(cfg.N_b // 2)
    x_b0 = torch.zeros(cfg.N_b // 2, 1, dtype=cfg.dtype, device=cfg.device)
    
    # Right boundary (x=1) - temperature must be zero for all time
    t_b1 = sample_uniform(cfg.N_b // 2)
    x_b1 = torch.ones(cfg.N_b // 2, 1, dtype=cfg.dtype, device=cfg.device)

    # Initial condition points (t=0) - starting temperature distribution
    x_0 = sample_uniform(cfg.N_0)  # Random x positions at t=0
    t_0 = torch.zeros(cfg.N_0, 1, dtype=cfg.dtype, device=cfg.device)

    return (x_f, t_f), (x_b0, t_b0), (x_b1, t_b1), (x_0, t_0)

# Training loop: This function trains the PINN with improved optimization strategy.
# Key improvements include weighted losses, learning rate scheduling, and better monitoring.
def train_loop(cfg):
    model = PINN(cfg).to(cfg.device)  # Initialize improved PINN architecture
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # Adam optimizer
    
    # Learning rate scheduler - gradually reduces learning rate for better convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    
    mse = nn.MSELoss()  # Mean squared error loss

    # Generate all training points using improved sampling strategy
    (x_f, t_f), (x_b0, t_b0), (x_b1, t_b1), (x_0, t_0) = make_training_sets(cfg)

    # Define target values for boundary and initial conditions
    u_b0_target = torch.zeros(len(t_b0), 1, dtype=cfg.dtype, device=cfg.device)  # u(0,t) = 0
    u_b1_target = torch.zeros(len(t_b1), 1, dtype=cfg.dtype, device=cfg.device)  # u(1,t) = 0
    u_0_target = torch.sin(torch.pi * x_0)  # u(x,0) = sin(π*x)

    # Track different loss components for monitoring training progress
    losses = {'total': [], 'pde': [], 'bc': [], 'ic': []}

    # Main training loop with improved loss balancing and monitoring
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()  # Reset gradients

        # PDE residual loss - how well the network satisfies the heat equation
        f = model.pde_residual(x_f, t_f, alpha=cfg.alpha)
        loss_f = mse(f, torch.zeros_like(f))

        # Boundary conditions loss - enforce u=0 at both ends of the rod
        u_b0 = model(torch.cat([x_b0, t_b0], dim=1))
        u_b1 = model(torch.cat([x_b1, t_b1], dim=1))
        loss_b = mse(u_b0, u_b0_target) + mse(u_b1, u_b1_target)

        # Initial condition loss - enforce u(x,0) = sin(π*x)
        u_0 = model(torch.cat([x_0, t_0], dim=1))
        loss_0 = mse(u_0, u_0_target)

        # Weighted total loss - boundary and initial conditions get much higher priority
        loss = cfg.w_f * loss_f + cfg.w_b * loss_b + cfg.w_0 * loss_0

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Store losses for analysis and monitoring
        losses['total'].append(loss.item())
        losses['pde'].append(loss_f.item())
        losses['bc'].append(loss_b.item())
        losses['ic'].append(loss_0.item())

        # Print progress every 1000 epochs with detailed loss breakdown
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Total={loss.item():.6f}, PDE={loss_f.item():.6f}, "
                  f"BC={loss_b.item():.6f}, IC={loss_0.item():.6f}")

    return model, losses

# Plot the training loss history to monitor convergence and identify potential issues.
# This helps us understand which components of the loss are dominating and whether
# the training is progressing well or getting stuck.
def plot_training_history(losses):
    plt.figure(figsize=(12, 4))
    
    # Plot total loss over time on log scale to see convergence
    plt.subplot(1, 2, 1)
    plt.semilogy(losses['total'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Total Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot individual loss components to see which constraints are being satisfied
    plt.subplot(1, 2, 2)
    plt.semilogy(losses['pde'], label='PDE Residual')
    plt.semilogy(losses['bc'], label='Boundary Conditions')
    plt.semilogy(losses['ic'], label='Initial Conditions')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Individual Loss Components')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualizes the PINN prediction vs exact solution at different time snapshots.
# This comprehensive visualization shows how well the network learned the physics
# by comparing predictions against the known analytical solution.
def plot_solution(model, alpha=1.0, nx=100, nt=100):
    # Create a grid of points for evaluation
    x = torch.linspace(0, 1, nx, dtype=cfg.dtype).unsqueeze(1)
    t = torch.linspace(0, 1, nt, dtype=cfg.dtype).unsqueeze(1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    # Get PINN predictions and exact solutions
    with torch.no_grad():
        u_pred = model(torch.cat([X_flat, T_flat], dim=1))
    U_pred = u_pred.reshape(nx, nt).cpu().numpy()
    U_exact = exact_solution(X, T, alpha).cpu().numpy()

    # Line plots at different time snapshots to show temporal evolution
    plt.figure(figsize=(15, 5))
    
    # Plot at t=0 (initial condition) - should match sin(π*x)
    plt.subplot(1, 3, 1)
    plt.plot(x.cpu().numpy(), U_exact[:, 0], 'b-', label="Exact at t=0", linewidth=2)
    plt.plot(x.cpu().numpy(), U_pred[:, 0], 'r--', label="PINN at t=0", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Solution at t=0 (Initial Condition)")
    plt.legend()
    plt.grid(True)

    # Plot at t=0.5 (middle time) - shows heat diffusion progress
    plt.subplot(1, 3, 2)
    mid_idx = nt // 2
    plt.plot(x.cpu().numpy(), U_exact[:, mid_idx], 'b-', label="Exact at t=0.5", linewidth=2)
    plt.plot(x.cpu().numpy(), U_pred[:, mid_idx], 'r--', label="PINN at t=0.5", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Solution at t=0.5")
    plt.legend()
    plt.grid(True)

    # Plot at t=1 (final time) - should be nearly zero due to heat dissipation
    plt.subplot(1, 3, 3)
    plt.plot(x.cpu().numpy(), U_exact[:, -1], 'b-', label="Exact at t=1", linewidth=2)
    plt.plot(x.cpu().numpy(), U_pred[:, -1], 'r--', label="PINN at t=1", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Solution at t=1")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Heatmaps for full space-time visualization of the solution evolution
    plt.figure(figsize=(15, 4))
    
    # Exact solution heatmap - the ground truth
    plt.subplot(1, 3, 1)
    plt.pcolormesh(T, X, U_exact, shading='auto')
    plt.colorbar()
    plt.title("Exact Solution u(x,t)")
    plt.xlabel("t")
    plt.ylabel("x")

    # PINN prediction heatmap - what our network learned
    plt.subplot(1, 3, 2)
    plt.pcolormesh(T, X, U_pred, shading='auto')
    plt.colorbar()
    plt.title("PINN Prediction u(x,t)")
    plt.xlabel("t")
    plt.ylabel("x")

    # Error heatmap - shows where the network made mistakes
    plt.subplot(1, 3, 3)
    error = np.abs(U_pred - U_exact)
    plt.pcolormesh(T, X, error, shading='auto')
    plt.colorbar()
    plt.title("Absolute Error |PINN - Exact|")
    plt.xlabel("t")
    plt.ylabel("x")

    plt.tight_layout()
    plt.show()
    
    # Print summary statistics about the error
    print(f"Max absolute error: {np.max(error):.6f}")
    print(f"Mean absolute error: {np.mean(error):.6f}")

# Computes a quantitative measure of error: L2 relative error
# This gives us a single number to assess overall accuracy of the PINN solution
def compute_error(model, alpha=1.0, nx=100, nt=100):
    x = torch.linspace(0, 1, nx, dtype=cfg.dtype).unsqueeze(1)
    t = torch.linspace(0, 1, nt, dtype=cfg.dtype).unsqueeze(1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(torch.cat([X_flat, T_flat], dim=1))
    U_pred = u_pred.reshape(nx, nt)
    U_exact = exact_solution(X, T, alpha)

    # L2 relative error: ||u_pred - u_exact|| / ||u_exact||
    error = torch.norm(U_pred - U_exact) / torch.norm(U_exact)
    return error.item()

# Main execution: train the improved model, visualize results, and assess accuracy
if __name__ == "__main__":
    print("Training improved PINN...")
    model, losses = train_loop(cfg)
    print("Training complete!")
    
    print("\nPlotting training history...")
    plot_training_history(losses)
    
    print("\nPlotting solution comparison...")
    plot_solution(model, alpha=cfg.alpha)
    
    l2_error = compute_error(model, alpha=cfg.alpha)
    print(f"\nL2 Relative Error: {l2_error:.6f}")