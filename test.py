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
    plt.savefig(f'solution for t ={t_slices}.png')  # Save the plot
    plt.show()
