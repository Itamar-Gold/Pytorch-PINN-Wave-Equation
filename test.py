import numpy as np
import torch
import matplotlib.pyplot as plt
from plot_functions import plot_slice as slice
from calculation_functions import numerical_solution as numerical_solution


def show_slices(model, x_bound: [float | int], t_slices: [float | int], n_points: int) -> None:

    model.eval()

    fig, axes = plt.subplots(len(t_slices), 2, figsize=(10, 4))

    x = np.linspace(x_bound[0], x_bound[1], n_points)
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    for index, t in enumerate(t_slices):
        T = t * np.ones(n_points)
        t_tensor = torch.tensor(T, dtype=torch.float32, requires_grad=True).unsqueeze(1)
        inputs = torch.cat([x_tensor, t_tensor], dim=1)
        pred = model.forward(inputs)
        u_pred, v_pred = pred[:, 0:1], pred[:, 1:2]
        u_pred = u_pred.detach().numpy()
        v_pred = v_pred.detach().numpy()
        u_solution, v_solution = numerical_solution(x, T)
        slice(T, x_bound, u_pred, u_solution, 'U(x,t)', axes[index, 0])
        slice(T, x_bound, v_pred, v_solution, 'V(x,t)', axes[index, 1])

    plt.tight_layout(pad=0.5)
    plt.savefig('images/solution for t = %s.png' % str(t_slices))  # Save the plot
    plt.show()


def error_l2norm(model, x, t):

    model.eval()

    u_true_tensor, v_true_tensor = numerical_solution(x, t)

    u_true_tensor = torch.tensor(u_true_tensor, dtype=torch.float32)
    v_true_tensor = torch.tensor(v_true_tensor, dtype=torch.float32)

    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    inputs = torch.cat([x_tensor, t_tensor], dim=1)

    pred = model.forward(inputs)
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]

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

    formatted_error_u = f"{relative_error_norm_u:.3f}%"
    formatted_error_v = f"{relative_error_norm_v:.3f}%"

    return formatted_error_u, formatted_error_v
