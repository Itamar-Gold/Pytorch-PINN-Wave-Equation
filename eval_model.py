import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Union
from plot_functions import plot_slice as custom_slice
from calculation_functions import numerical_solution

# Define device dynamically for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_slices(model, x_bound: List[Union[float, int]], t_slices: List[Union[float, int]], n_points: int) -> None:
    """
    Plots slices of the predicted solution versus the analytical numerical solution.
    """
    model.to(device)
    model.eval()

    fig, axes = plt.subplots(len(t_slices), 2, figsize=(10, 4))
    
    # Ensure axes is always 2D even if there is only 1 time slice
    if len(t_slices) == 1:
        axes = np.expand_dims(axes, axis=0)

    # Create spatial grid
    x = np.linspace(x_bound[0], x_bound[1], n_points)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)

    # Disable gradient calculation for testing/plotting to save memory and compute
    with torch.no_grad():
        for index, t in enumerate(t_slices):
            T = t * np.ones(n_points)
            t_tensor = torch.tensor(T, dtype=torch.float32, device=device).unsqueeze(1)
            inputs = torch.cat([x_tensor, t_tensor], dim=1)
            
            # Predict
            pred = model.forward(inputs)
            u_pred, v_pred = pred[:, 0:1], pred[:, 1:2]
            
            # Move back to CPU for NumPy plotting
            u_pred = u_pred.cpu().numpy()
            v_pred = v_pred.cpu().numpy()
            
            # Analytical Solution
            u_solution, v_solution = numerical_solution(x, T)
            
            # Use custom slice plotting logic
            custom_slice(T, x_bound, u_pred, u_solution, 'U(x,t)', axes[index, 0])
            custom_slice(T, x_bound, v_pred, v_solution, 'V(x,t)', axes[index, 1])

    plt.tight_layout(pad=0.5)
    
    # Ensure directory exists before saving
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/solution_for_t_%s.png' % str(t_slices).replace(', ', '_').replace('.', 'p'))
    plt.show()


def error_l2norm(model, x, t):
    """
    Computes the L2 relative error norm between the model predictions and the exact 
    analytical solution to evaluate model accuracy.
    """
    model.to(device)
    model.eval()

    # Get True Analytical Solutions
    u_true, v_true = numerical_solution(x, t)

    u_true_tensor = torch.tensor(u_true, dtype=torch.float32, device=device)
    v_true_tensor = torch.tensor(v_true, dtype=torch.float32, device=device)

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)
    inputs = torch.cat([x_tensor, t_tensor], dim=1)

    # Disable gradients for testing to save computation
    with torch.no_grad():
        pred = model.forward(inputs)
        
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]

    # Calculate difference vectors
    error_vec_u = u_true_tensor - u_pred.squeeze(-1)
    error_vec_v = v_true_tensor - v_pred.squeeze(-1)

    # Compute the L2 norms of the error vectors
    error_norm_u = torch.norm(error_vec_u, p=2)
    true_norm_u = torch.norm(u_true_tensor, p=2)

    error_norm_v = torch.norm(error_vec_v, p=2)
    true_norm_v = torch.norm(v_true_tensor, p=2)

    # Calculate the relative L2 norms of the errors and convert to percentage
    relative_error_norm_u = 100 * (error_norm_u / true_norm_u)
    relative_error_norm_v = 100 * (error_norm_v / true_norm_v)

    # Format output as string
    formatted_error_u = f"{relative_error_norm_u.item():.3f}%"
    formatted_error_v = f"{relative_error_norm_v.item():.3f}%"

    return formatted_error_u, formatted_error_v
