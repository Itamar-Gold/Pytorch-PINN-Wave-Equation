import numpy as np
import torch.autograd as autograd
import torch
from scipy.stats import qmc


def grad_calculation_second(input, derivative_by_x, derivative_by_t):
    # Compute first derivatives
    u_t = autograd.grad(input, derivative_by_t, torch.ones_like(input), create_graph=True)[0]  # derivative with respect to t
    u_x = autograd.grad(input, derivative_by_x, torch.ones_like(input), create_graph=True)[0]  # derivative with respect to x
    # Compute second derivatives
    u_tt = autograd.grad(u_t, derivative_by_t, torch.ones_like(u_t), create_graph=True)[0]  # second derivative with respect to t
    u_xx = autograd.grad(u_x, derivative_by_x, torch.ones_like(u_x), create_graph=True)[0]  # second derivative with respect to x
    return u_xx, u_tt


def grad_calculation_first(input, derivative_by_x, derivative_by_t):
    # Compute first derivatives
    u_t = autograd.grad(input, derivative_by_t, torch.ones_like(input), create_graph=True)[0]  # derivative with respect to t
    u_x = autograd.grad(input, derivative_by_x, torch.ones_like(input), create_graph=True)[0]  # derivative with respect to x
    return u_x, u_t


def data_sampler(N, x_bound, t_bound):
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=N)
    l_bounds = [x_bound[0], t_bound[0]]
    u_bounds = [x_bound[1], t_bound[1]]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    x = sample_scaled[:, 0]
    t = sample_scaled[:, 1]
    return x, t


def numerical_solution(x, t):
    # Solve the Helmholtz equation
    U = np.cos(np.pi * t / 4) * np.sin(np.pi * (x + 1) / 2)
    V = np.sin(np.pi * t / 4) * np.sin(np.pi * (x + 1) / 2)
    return U, V


def prep_anim(model, x, t):

    model.eval()
    x_mesh, t_mesh = np.meshgrid(x, t)

    real_u, real_v = numerical_solution(x_mesh, t_mesh)
    real_h = np.sqrt(real_u**2 + real_v**2)

    x_mesh = x_mesh.flatten()
    t_mesh = t_mesh.flatten()

    x_tensor = torch.tensor(x_mesh, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_tensor = torch.tensor(t_mesh, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    inputs = torch.cat([x_tensor, t_tensor], dim=1)

    pred = model.forward(inputs)
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]

    u_pred = u_pred.detach().numpy()
    v_pred = v_pred.detach().numpy()

    u_pred = np.reshape(u_pred, (len(x), len(x)))
    v_pred = np.reshape(v_pred, (len(x), len(x)))

    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    model.train()

    return u_pred, real_u, v_pred, real_v, h_pred, real_h
