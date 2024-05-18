import matplotlib.pyplot as plt
import torch.nn as nn
import time
import torch
import calculation_functions as cf
from plot_functions import animate_3d_result as animate


class PINN(nn.Module):
    """
    Defines a fully connected neural network with customizable depth and width.
    """
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super(PINN, self).__init__()
        activation = nn.Tanh

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

    ic_pred = model(train_data)
    u_icp = ic_pred[:, 0:1]  # Takes all rows of the first column and retains the column dimension
    v_icp = ic_pred[:, 1:2]

    u_x, u_t = cf.grad_calculation_first(u_icp, x, t)
    # detaching x & t for the equation calculation
    x = x.detach().numpy()
    t = t.detach().numpy()
    # solving numerically for u and v
    u_ic, v_ic = cf.numerical_solution(x, t)
    # returning from numpy to tensor
    u_ic = torch.tensor(u_ic, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    v_ic = torch.tensor(v_ic, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    # calculating the loss for 1. u(x,t=0)  2. v(x,t=0)  3.  u_t(x,t=0)=v(x,t=0)
    initial_condition_loss = torch.mean((u_icp - u_ic) ** 2 + (v_icp - v_ic) ** 2 + (u_t - v_ic) ** 2)

    return initial_condition_loss


def mse_loss_b(model, x, t, train_data):

    # can be used to calculate the loss of the model's boundary and initial conditions compliance
    b_pred = model(train_data)

    x = x.detach().numpy()
    t = t.detach().numpy()

    u_b, v_b = cf.numerical_solution(x, t)

    u_b = torch.tensor(u_b, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    v_b = torch.tensor(v_b, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    u_bp = b_pred[:, 0:1]
    v_bp = b_pred[:, 1:2]

    boundary_loss = torch.mean((u_bp - u_b) ** 2 + (v_bp - v_b) ** 2)

    return boundary_loss


def mse_loss_p(model, x, t, train_data):
    # can be used to calculate the loss of the model's physics compliance
    # x_physics = torch.linspace(-1,1,100).view(-1,1).requires_grad_(True)
    # t_physics = torch.linspace(0,4,100).view(-1,1).requires_grad_(True)
    # physics_inputs = torch.cat([x_physics, t_physics], dim=1)

    p_pred = model(train_data)

    u_p = p_pred[:, 0:1]
    v_p = p_pred[:, 1:2]

    u_p.requires_grad_(True)
    v_p.requires_grad_(True)

    u_xx, u_tt = cf.grad_calculation_second(u_p, x, t)
    v_xx, v_tt = cf.grad_calculation_second(v_p, x, t)

    # Wave equation residual
    c = 1
    u_physics = u_tt - c ** 2 * u_xx
    v_physics = v_tt - c ** 2 * v_xx

    physics_loss = torch.mean(u_physics ** 2 + v_physics ** 2)

    return physics_loss


def mse_loss(model, lamda1, lamda2, lamda3, n_0, n_b, n_p):

    # the initial condition loss
    # Initial conditions t=0 for all x
    xic_1, tic_1 = cf.data_sampler(n_0, [-1, 1], [0, 0.0000001])  # sample between -1 < x < 1 and t = 0
    x_ic = torch.tensor(xic_1, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_ic = torch.tensor(tic_1, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    ic_inputs = torch.cat([x_ic, t_ic], dim=1)

    ic_loss = mse_loss_0(model, x_ic, t_ic, ic_inputs)
    # end of initial condition loss

    # the first boundary condition loss
    # Boundary for x = -1 for all t
    xbc_1, tbc_1 = cf.data_sampler(n_b, [-1.00000, -0.999999], [0, 4]) # sample x = 0 and 0 < t < 4
    x_boundary = torch.tensor(xbc_1, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_boundary = torch.tensor(tbc_1, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    boundary_inputs = torch.cat([x_boundary, t_boundary], dim=1)
    f_boundary_loss = mse_loss_b(model, x_boundary, t_boundary, boundary_inputs)
    # end of first boundary condition loss

    # the second boundary condition loss
    # Boundary for x = 1 for all t
    xbc_2, tbc_2 = cf.data_sampler(n_b, [0.999999, 1.00000], [0, 4]) # sample x = 1 and 0 < t < 4
    x_boundary2 = torch.tensor(xbc_2, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_boundary2 = torch.tensor(tbc_2, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    boundary_inputs2 = torch.cat([x_boundary2, t_boundary2], dim=1)
    s_boundary_loss = mse_loss_b(model, x_boundary2, t_boundary2, boundary_inputs2)
    # end of second boundary condition loss

    # the physics loss
    x_physics, t_physics = cf.data_sampler(n_p, [-1, 1], [0, 4]) # sample between -1 < x < 1 and 0 < t < 4
    x_physics = torch.tensor(x_physics, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_physics = torch.tensor(t_physics, dtype=torch.float32, requires_grad=True).unsqueeze(1)

    physics_inputs = torch.cat([x_physics, t_physics], dim=1)

    physics_loss = mse_loss_p(model, x_physics, t_physics, physics_inputs)
    # end of physics loss

    # combining the first and second boundary loss
    total_boundary_loss = torch.mean(f_boundary_loss + s_boundary_loss)

    # total loss
    loss_ = lamda1*ic_loss + lamda2*total_boundary_loss + lamda3*physics_loss

    return loss_


def train_model(model, optimizer, inputs, x, t, animation_prep=False):

    start_time = time.time()

    u, v = cf.numerical_solution(x, t)
    u_tensor = torch.tensor(u, dtype=torch.float32)
    v_tensor = torch.tensor(v, dtype=torch.float32)
    # Keep track of the loss
    iters = []
    files = []
    losses  = []

    # Training loop
    for i in range(4000):

        optimizer.zero_grad()
        # Model prediction
        predictions = model(inputs)
        um, vm = predictions[:, 0], predictions[:, 1]

        # Calculate the Data loss
        model_loss = torch.mean((um - u_tensor)**2) + torch.mean((vm - v_tensor)**2)

        # Calculate the physics-based loss
        eq_loss = mse_loss(model, 0.01, 0.01, 0.001, 100, 100, 500)
        # lamda1*ic_loss + lamda2*total_boundary_loss + lamda3*physics_loss  ,N_0, N_b, N_f

        loss = model_loss + eq_loss
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')
        # Loss for plots
        if i % 100 == 0:
            iters.append(i)
            losses.append(loss.item())
        if animation_prep:
            if i % 40 == 0:

                u_pred, real_u, v_pred, real_v, h_pred, real_h = cf.prep_anim(model, x, t)
                animate(x, t, u_pred, 'u(x,t)', i, real_u)
                animate(x, t, v_pred, 'v(x,t)', i, real_v)
                animate(x, t, h_pred, 'h(x,t)', i, real_h)

    elapsed = time.time() - start_time
    print('Training time: %.2f' % elapsed)

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/loss_curve.png')  # Save the loss curve plot
    plt.show()

    return iters, losses
