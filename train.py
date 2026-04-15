import matplotlib.pyplot as plt
import torch.nn as nn
import time
import torch
import numpy as np
import calculation_functions as cf
from plot_functions import animate_3d_result as animate

# Define device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SineActivation(nn.Module):
    """
    Sine activation function for building SIREN (Sinusoidal Representation Networks).
    Helps overcome the 'spectral bias' of standard activations like Tanh/ReLU, 
    allowing the network to capture high-frequency components and complex derivatives.
    """
    def forward(self, x):
        return torch.sin(x)


class PINN(nn.Module):
    """
    Defines a fully connected neural network with customizable depth and width.
    """
    def __init__(self, n_input, n_output, n_hidden, n_layers, use_siren=True):
        super(PINN, self).__init__()
        
        # Use SIREN by default to solve spectral bias
        activation = SineActivation if use_siren else nn.Tanh

        # Input layer
        layers = [nn.Linear(n_input, n_hidden), activation()]

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation())

        # Output layer
        self.model = nn.Sequential(*layers, nn.Linear(n_hidden, n_output))

    def forward(self, x):
        return self.model(x)


def mse_loss_0(model, x, t, train_data):
    """
    Calculates the Initial Condition loss.
    Loss on u(x,t=0), v(x,t=0), and u_t(x,t=0)=v(x,t=0)
    """
    ic_pred = model(train_data)
    u_icp = ic_pred[:, 0:1]  # Predicted u
    v_icp = ic_pred[:, 1:2]  # Predicted v

    u_x, u_t = cf.grad_calculation_first(u_icp, x, t)

    # Calculate exact initial conditions purely in PyTorch.
    # Detach targets to ensure they don't break the computation graph during backprop
    with torch.no_grad():
        u_ic, v_ic = cf.numerical_solution_pt(x, t)

    # Calculate MSE for initial conditions
    initial_condition_loss = torch.mean((u_icp - u_ic) ** 2 + (v_icp - v_ic) ** 2 + (u_t - v_ic) ** 2)

    return initial_condition_loss


def mse_loss_b(model, x, t, train_data):
    """
    Calculates the Boundary Condition loss.
    """
    b_pred = model(train_data)
    
    u_bp = b_pred[:, 0:1]
    v_bp = b_pred[:, 1:2]

    # Calculate analytical solutions at the boundaries. Detach targets.
    with torch.no_grad():
        u_b, v_b = cf.numerical_solution_pt(x, t)

    boundary_loss = torch.mean((u_bp - u_b) ** 2 + (v_bp - v_b) ** 2)

    return boundary_loss


def mse_loss_p(model, x, t, train_data):
    """
    Calculates the Physics-Informed loss based on the wave equation residual.
    Residual = u_tt - c^2 * u_xx = 0
    """
    p_pred = model(train_data)

    u_p = p_pred[:, 0:1]
    v_p = p_pred[:, 1:2]

    u_xx, u_tt = cf.grad_calculation_second(u_p, x, t)
    v_xx, v_tt = cf.grad_calculation_second(v_p, x, t)

    # Wave equation residual
    c = 1.0
    u_physics = u_tt - (c ** 2) * u_xx
    v_physics = v_tt - (c ** 2) * v_xx

    # Calculate MSE of the residuals
    physics_loss = torch.mean(u_physics ** 2 + v_physics ** 2)

    return physics_loss


def mse_loss(model, weight1, weight2, weight3, ic_data, bc1_data, bc2_data, phys_data):
    """
    Aggregates all components of the loss function. Data points are passed in 
    and detached -> require_grad to ensure fresh computational graphs each iteration.
    """

    x_ic, t_ic = ic_data
    # Clone and require grad inside the loop to avoid PyTorch graph retention errors!
    x_ic = x_ic.clone().detach().requires_grad_(True)
    t_ic = t_ic.clone().detach().requires_grad_(True)
    ic_inputs = torch.cat([x_ic, t_ic], dim=1)
    ic_loss = mse_loss_0(model, x_ic, t_ic, ic_inputs)

    x_bc1, t_bc1 = bc1_data
    x_bc1 = x_bc1.clone().detach().requires_grad_(True)
    t_bc1 = t_bc1.clone().detach().requires_grad_(True)
    bc1_inputs = torch.cat([x_bc1, t_bc1], dim=1)
    f_boundary_loss = mse_loss_b(model, x_bc1, t_bc1, bc1_inputs)

    x_bc2, t_bc2 = bc2_data
    x_bc2 = x_bc2.clone().detach().requires_grad_(True)
    t_bc2 = t_bc2.clone().detach().requires_grad_(True)
    bc2_inputs = torch.cat([x_bc2, t_bc2], dim=1)
    s_boundary_loss = mse_loss_b(model, x_bc2, t_bc2, bc2_inputs)

    x_physics, t_physics = phys_data
    x_physics = x_physics.clone().detach().requires_grad_(True)
    t_physics = t_physics.clone().detach().requires_grad_(True)
    physics_inputs = torch.cat([x_physics, t_physics], dim=1)
    physics_loss = mse_loss_p(model, x_physics, t_physics, physics_inputs)

    # Combine the first and second boundary loss
    total_boundary_loss = torch.mean(f_boundary_loss + s_boundary_loss)

    # Total weighted loss
    loss_ = weight1 * ic_loss + weight2 * total_boundary_loss + weight3 * physics_loss

    return loss_, ic_loss, total_boundary_loss, physics_loss


def create_dataset(n_points_multiplier=10, base_n_points=(100, 100, 500)):
    """
    Creates a massive dataset of collocation points to prevent overfitting.
    We will sample batches from this dataset during training.
    """
    n_ic_base, n_bc_base, n_phys_base = base_n_points
    
    large_n_points = (
        n_ic_base * n_points_multiplier, 
        n_bc_base * n_points_multiplier, 
        n_phys_base * n_points_multiplier
    )
    
    # Sample points
    xic, tic = cf.data_sampler(large_n_points[0], [-1, 1], [0, 0.0000001])
    xbc_1, tbc_1 = cf.data_sampler(large_n_points[1], [-1.00000, -0.999999], [0, 4])
    xbc_2, tbc_2 = cf.data_sampler(large_n_points[1], [0.999999, 1.00000], [0, 4])
    x_p, t_p = cf.data_sampler(large_n_points[2], [-1, 1], [0, 4])
    
    # Store in a dictionary on the target device
    dataset = {
        'ic_x': torch.tensor(xic, dtype=torch.float32, device=device).unsqueeze(1), 
        'ic_t': torch.tensor(tic, dtype=torch.float32, device=device).unsqueeze(1),
        'bc1_x': torch.tensor(xbc_1, dtype=torch.float32, device=device).unsqueeze(1), 
        'bc1_t': torch.tensor(tbc_1, dtype=torch.float32, device=device).unsqueeze(1),
        'bc2_x': torch.tensor(xbc_2, dtype=torch.float32, device=device).unsqueeze(1), 
        'bc2_t': torch.tensor(tbc_2, dtype=torch.float32, device=device).unsqueeze(1),
        'phys_x': torch.tensor(x_p, dtype=torch.float32, device=device).unsqueeze(1), 
        'phys_t': torch.tensor(t_p, dtype=torch.float32, device=device).unsqueeze(1)
    }
    
    return dataset


def get_batch(dataset, batch_sizes):
    """
    Randomly samples a batch from the large pre-computed dataset.
    """
    n_ic, n_bc, n_phys = batch_sizes
    
    # Randomly select indices for each boundary/physics condition
    ic_idx = torch.randperm(dataset['ic_x'].size(0))[:n_ic]
    bc1_idx = torch.randperm(dataset['bc1_x'].size(0))[:n_bc]
    bc2_idx = torch.randperm(dataset['bc2_x'].size(0))[:n_bc]
    phys_idx = torch.randperm(dataset['phys_x'].size(0))[:n_phys]
    
    # Extract the batch data
    ic_data = (dataset['ic_x'][ic_idx], dataset['ic_t'][ic_idx])
    bc1_data = (dataset['bc1_x'][bc1_idx], dataset['bc1_t'][bc1_idx])
    bc2_data = (dataset['bc2_x'][bc2_idx], dataset['bc2_t'][bc2_idx])
    phys_data = (dataset['phys_x'][phys_idx], dataset['phys_t'][phys_idx])
    
    return ic_data, bc1_data, bc2_data, phys_data


def train_model(model, optimizer, scheduler, inputs, epochs, weights, batch_sizes, dataset, animation_prep=False):
    """
    Main training loop using batched sampling to prevent overfitting.
    """
    start_time = time.time()
    model.to(device)
    inputs = inputs.to(device)

    # Data loss calculation targets - detached to ensure they don't break the graph
    with torch.no_grad():
        u, v = cf.numerical_solution_pt(inputs[:, 0:1], inputs[:, 1:2])
        u_tensor = u.squeeze(-1).to(device)
        v_tensor = v.squeeze(-1).to(device)

    weight_ic, weight_bc, weight_phys = weights

    iters = []
    losses = []
    ic_losses = []
    bc_losses = []
    phy_losses = []
    model_losses = []

    # Training loop
    for i in range(epochs):
        optimizer.zero_grad()
        
        # Get a fresh random batch every epoch
        ic_batch, bc1_batch, bc2_batch, phys_batch = get_batch(dataset, batch_sizes)
        
        # Model prediction on regular grid
        predictions = model(inputs)
        um, vm = predictions[:, 0], predictions[:, 1]

        # Calculate the Data loss (supervised learning component)
        model_loss = torch.mean((um - u_tensor)**2) + torch.mean((vm - v_tensor)**2)

        # Calculate the physics-based loss (unsupervised learning component)
        eq_loss, ic_loss, bc_loss, phy_loss = mse_loss(model, weight_ic, weight_bc, weight_phys,
                           ic_batch, bc1_batch, bc2_batch, phys_batch)

        loss = model_loss + eq_loss
        loss.backward()
        optimizer.step()
        
        # Step the learning rate scheduler
        if scheduler:
            scheduler.step()

        if i % 200 == 0:
            print(f'Iteration {i}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6e}')
            
        # Record loss for plotting
        if i % 100 == 0:
            iters.append(i)
            losses.append(loss.item())
            phy_losses.append(phy_loss.item())
            ic_losses.append(ic_loss.item())
            bc_losses.append(bc_loss.item())
            model_losses.append(model_loss.item())
            
        # Create frames for animation
        if animation_prep and i % 20 == 0:
            x_grid = np.linspace(-1, 1, 100)
            t_grid = np.linspace(0, 4, 100)
            u_pred, real_u, v_pred, real_v, h_pred, real_h = cf.prep_anim(model, x_grid, t_grid)
            animate(x_grid.tolist(), t_grid.tolist(), u_pred, 'u(x,t)', i, real_u)
            animate(x_grid.tolist(), t_grid.tolist(), v_pred, 'v(x,t)', i, real_v)
            animate(x_grid.tolist(), t_grid.tolist(), h_pred, 'h(x,t)', i, real_h)

    elapsed = time.time() - start_time
    print('Training time: %.2f seconds' % elapsed)

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, label='Total Loss')
    plt.plot(range(1, len(phy_losses) + 1), phy_losses, label='Physics Loss')
    plt.plot(range(1, len(ic_losses) + 1), ic_losses, label='IC Loss')
    plt.plot(range(1, len(bc_losses) + 1), bc_losses, label='BC Loss')
    plt.plot(range(1, len(model_losses) + 1), model_losses, label='Model Loss')

    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.yscale('log') # Log scale is much better for viewing PINN convergence
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/loss_curve.png')
    # plt.show()

    return iters, losses
