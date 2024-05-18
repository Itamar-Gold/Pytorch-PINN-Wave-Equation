import matplotlib.pyplot as plt
import numpy as np


def animate_3d_result(x: [float], t: [float], predicted_solution: [float],
                      solution_title: [str], iteration: int, true_solution=None) -> None:

    x_mesh, t_mesh = np.meshgrid(x, t)
    fig = plt.figure(figsize=(14, 7))

    # If the true surface is provided, plot it
    if true_solution is not None:
        ax_true = fig.add_subplot(122, projection='3d')
        surf_true = ax_true.plot_surface(x_mesh, t_mesh, true_solution, cmap='plasma', edgecolor='none')
        ax_true.set_title(f'True Solution')
        ax_true.set_xlabel('Spatial domain (x)')
        ax_true.set_ylabel('Temporal domain (t)')
        ax_true.set_zlabel(solution_title)
        ax_true.xaxis.set_major_locator(plt.MaxNLocator(5))  # correct the axis for the plots
        ax_true.yaxis.set_major_locator(plt.MaxNLocator(5))
        fig.colorbar(surf_true, shrink=0.5, aspect=10, ax=ax_true, location='left')
        ax_true.view_init(elev=10, azim=40)  # Adjust the azim parameter here for better view

        # Plot predicted wave
        ax_pred = fig.add_subplot(121, projection='3d')
        surf_pred = ax_pred.plot_surface(x_mesh, t_mesh, predicted_solution, cmap='plasma', edgecolor='none')
        ax_pred.set_title(f'Predicted Solution  iteration = {iteration}')
        ax_pred.set_xlabel('Spatial domain (x)')
        ax_pred.set_ylabel('Temporal domain (t)')
        ax_pred.set_zlabel(solution_title)
        ax_pred.xaxis.set_major_locator(plt.MaxNLocator(5))  # correct the axis for the plots
        ax_pred.yaxis.set_major_locator(plt.MaxNLocator(5))
        fig.colorbar(surf_pred, shrink=0.5, aspect=10, ax=ax_pred, location='left')
        ax_pred.view_init(elev=10, azim=40)  # Adjust the azim parameter here for better view

    else:
        # Plot predicted wave
        ax_pred = fig.add_subplot(111, projection='3d')
        surf_pred = ax_pred.plot_surface(x_mesh, t_mesh, predicted_solution, cmap='plasma', edgecolor='none')
        ax_pred.set_title(f'Predicted Solution   iteration = {iteration}')
        ax_pred.set_xlabel('Spatial domain (x)')
        ax_pred.set_ylabel('Temporal domain (t)')
        ax_pred.set_zlabel(solution_title)
        ax_pred.xaxis.set_major_locator(plt.MaxNLocator(5))  # correct the axis for the plots
        ax_pred.yaxis.set_major_locator(plt.MaxNLocator(5))
        fig.colorbar(surf_pred, shrink=0.5, aspect=10, ax=ax_pred, location='left')
        ax_pred.view_init(elev=10, azim=40)  # Adjust the azim parameter here for better view

    plt.savefig(f'images/{solution_title}/training_{iteration}.png')
    plt.close()


def plot_slice(t_time: list[float | int], x_bound: [float | int, float | int],
               prediction: list[float], solution: list[float], variable_name: str, ax) -> None:
    """
    plot and visualize a slice of the solution for a specific time
    :param t_time: a list of the same time (like: [1, 1 ... 1, 1])
    :param x_bound: a bound of the x domain ([-1, 1])
    :param prediction: the output prediction of the model
    :param solution: the exact solution from the equation
    :param variable_name: a string of the function value ('H(x,t)' / 'U(x,t)'/ 'V(x,t)')
    :param ax: a pointer to a subplot in a figure
    :return:

    Example:
    this example plot 2 slices of 2 different times (3 and 4):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_slice(t1, x_bound, prediction1, solution1, variable_name1, axes[0])
    plot_slice(t2, x_bound, prediction2, solution2, variable_name2, axes[1])

    plt.tight_layout()
    plt.show()
    """

    x_domain = np.linspace(x_bound[0], x_bound[1], 100)
    ax.set_title(f'{variable_name} test for t = {t_time[0]}')
    ax.plot(x_domain, prediction, label=f'{variable_name} PINN')
    ax.plot(x_domain, solution, label=f'{variable_name} equation')
    ax.set_xlabel("x")
    ax.set_ylabel(f'{variable_name} value')
    ax.legend()
    ax.grid(True)
