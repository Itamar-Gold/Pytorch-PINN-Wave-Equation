import numpy as np
import torch.autograd as autograd
import torch
from scipy.stats import qmc

# Define device for GPU acceleration if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def grad_calculation_second(input_var, derivative_by_x, derivative_by_t):
    """
    Calculates the second-order derivatives of the network's output with respect to inputs x and t.
    """
    # Compute first derivatives
    u_t = autograd.grad(input_var, derivative_by_t, torch.ones_like(input_var), create_graph=True)[0]  # derivative with respect to t
    u_x = autograd.grad(input_var, derivative_by_x, torch.ones_like(input_var), create_graph=True)[0]  # derivative with respect to x
    
    # Compute second derivatives
    u_tt = autograd.grad(u_t, derivative_by_t, torch.ones_like(u_t), create_graph=True)[0]  # second derivative with respect to t
    u_xx = autograd.grad(u_x, derivative_by_x, torch.ones_like(u_x), create_graph=True)[0]  # second derivative with respect to x
    
    return u_xx, u_tt


def grad_calculation_first(input_var, derivative_by_x, derivative_by_t):
    """
    Calculates the first-order derivatives of the network's output with respect to inputs x and t.
    """
    # Compute first derivatives
    u_t = autograd.grad(input_var, derivative_by_t, torch.ones_like(input_var), create_graph=True)[0]  # derivative with respect to t
    u_x = autograd.grad(input_var, derivative_by_x, torch.ones_like(input_var), create_graph=True)[0]  # derivative with respect to x
    
    return u_x, u_t


def data_sampler(N, x_bound, t_bound):
    """
    Uses Latin Hypercube Sampling to sample collocation points across the domain.
    This provides a more uniform coverage of the parameter space than standard random sampling.
    """
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=N)
    l_bounds = [x_bound[0], t_bound[0]]
    u_bounds = [x_bound[1], t_bound[1]]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    x = sample_scaled[:, 0]
    t = sample_scaled[:, 1]
    return x, t


def numerical_solution(x, t):
    """
    NumPy implementation of the exact solution, used for evaluation and plotting.
    """
    U = np.cos(np.pi * t / 4) * np.sin(np.pi * (x + 1) / 2)
    V = np.sin(np.pi * t / 4) * np.sin(np.pi * (x + 1) / 2)
    return U, V


def numerical_solution_pt(x, t):
    """
    PyTorch implementation of the exact solution to be used directly in loss calculations.
    This avoids expensive CPU/GPU memory transfers and detachment from the computation graph.
    Uses np.pi instead of torch.pi to maintain backwards compatibility with older PyTorch versions.
    """
    U = torch.cos(np.pi * t / 4.0) * torch.sin(np.pi * (x + 1.0) / 2.0)
    V = torch.sin(np.pi * t / 4.0) * torch.sin(np.pi * (x + 1.0) / 2.0)
    return U, V


def prep_anim(model, x, t):
    """
    Prepares prediction vs analytical data for visualization over time.
    """
    model.eval()
    x_mesh, t_mesh = np.meshgrid(x, t)

    real_u, real_v = numerical_solution(x_mesh, t_mesh)
    real_h = np.sqrt(real_u**2 + real_v**2)

    x_mesh_flat = x_mesh.flatten()
    t_mesh_flat = t_mesh.flatten()

    x_tensor = torch.tensor(x_mesh_flat, dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor = torch.tensor(t_mesh_flat, dtype=torch.float32).unsqueeze(1).to(device)

    inputs = torch.cat([x_tensor, t_tensor], dim=1)

    # Disable gradient tracking for inference
    with torch.no_grad():
        pred = model(inputs)
    
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]

    # Move back to CPU for numpy operations
    u_pred = u_pred.cpu().numpy()
    v_pred = v_pred.cpu().numpy()

    u_pred = np.reshape(u_pred, (len(t), len(x)))
    v_pred = np.reshape(v_pred, (len(t), len(x)))

    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    model.train()

    return u_pred, real_u, v_pred, real_v, h_pred, real_h
