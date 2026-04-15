import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

import calculation_functions as cf
from train import PINN, mse_loss, create_dataset, get_batch
import eval_model

# Setup device configuration globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_for_hyperparam_comparison(model, dataset, epochs=4000, lr=1e-3, decay=0.999, batch=100, model_name="Model"):
    """
    A lightweight training loop specifically designed for benchmarking hyperparameters.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = ExponentialLR(optimizer, gamma=decay)
    
    batch_sizes = (batch, batch, batch * 5)
    weights = (0.01, 0.01, 0.001)
    weight_ic, weight_bc, weight_phys = weights

    # Supervised data targets for overall loss
    x, t = cf.data_sampler(100, [-1, 1], [0, 4])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
    inputs = torch.cat([x_tensor, t_tensor], dim=1)

    with torch.no_grad():
        u, v = cf.numerical_solution_pt(inputs[:, 0:1], inputs[:, 1:2])
        u_tensor = u.squeeze(-1).to(device)
        v_tensor = v.squeeze(-1).to(device)

    history = {'total': [], 'physics': [], 'ic': [], 'bc': []}
    start_time = time.time()
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        # Get a fresh random batch
        ic_batch, bc1_batch, bc2_batch, phys_batch = get_batch(dataset, batch_sizes)
        
        # Model prediction
        predictions = model(inputs)
        um, vm = predictions[:, 0], predictions[:, 1]
        model_loss = torch.mean((um - u_tensor)**2) + torch.mean((vm - v_tensor)**2)

        # Calculate losses
        eq_loss, ic_loss, bc_loss, phy_loss = mse_loss(model, weight_ic, weight_bc, weight_phys,
                           ic_batch, bc1_batch, bc2_batch, phys_batch)

        loss = model_loss + eq_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            history['total'].append(loss.item())
            history['physics'].append(phy_loss.item())
            history['ic'].append(ic_loss.item())
            history['bc'].append(bc_loss.item())
            
        if i % 500 == 0:
            print(f"[{model_name}] Epoch {i:4d} | Total Loss: {loss.item():.6e} | Physics Loss: {phy_loss.item():.6e}")

    print(f"[{model_name}] Training completed in {time.time() - start_time:.2f} seconds")
    
    # Calculate final L2 error
    x_test, t_test = cf.data_sampler(2500, [-1, 1], [0, 4])
    u_error, v_error = eval_model.error_l2norm(model, x_test, t_test)
    print(f"[{model_name}] L2 Error -> U: {u_error} | V: {v_error}\n")
    
    return history, (u_error, v_error)


def main():
    print(f"Using device: {device}\n")
    epochs = 4000
    
    print("Creating large underlying dataset (10x normal size) for batch sampling...")
    dataset = create_dataset(n_points_multiplier=10)
    print("Dataset created successfully.\n")

    # Define hyperparameter configurations to test
    configs = [
        {"name": "LR 5e-3", "use_siren": True, "batch": 50, "lr": 5e-3, "decay": 0.999, "color": "blue", "ls": "--"},
        {"name": "LR 3e-3", "use_siren": True, "batch": 50, "lr": 3e-3, "decay": 0.999, "color": "purple", "ls": "-"},
        {"name": "LR 1e-3", "use_siren": True, "batch": 50, "lr": 1e-3, "decay": 0.999, "color": "red", "ls": "-"},
        {"name": "LR 8e-4", "use_siren": True, "batch": 50, "lr": 8e-4, "decay": 0.999, "color": "green", "ls": "-"},
        {"name": "LR 6e-4", "use_siren": True, "batch": 50, "lr": 6e-4, "decay": 0.999, "color": "orange", "ls": "-"},
    ]
    
    results = {}

    for config in configs:
        print("======================================================")
        print(f"   Training: {config['name']}")
        print("======================================================")
        
        # Initialize model
        model = PINN(n_input=2, n_output=2, n_hidden=32, n_layers=8, use_siren=config['use_siren'])
        
        # Train and record
        history, final_errors = train_for_hyperparam_comparison(
            model, dataset, epochs=epochs, lr=config['lr'], decay=config['decay'], batch=config['batch'], model_name=config['name']
        )
        
        results[config['name']] = {
            'history': history,
            'errors': final_errors,
            'color': config['color'],
            'ls': config['ls']
        }

    # ---------------------------------------------------------
    # Plotting the Comparisons
    # ---------------------------------------------------------
    epochs_range = range(0, epochs, 100)
    
    plt.figure(figsize=(18, 5))
    
    # 1. Total Loss Comparison
    plt.subplot(1, 3, 1)
    for name, data in results.items():
        plt.plot(epochs_range, data['history']['total'], label=f"{name}\nU: {data['errors'][0]}, V: {data['errors'][1]}", 
                 color=data['color'], linestyle=data['ls'], linewidth=2)
    plt.yscale('log')
    plt.title('Total Loss Convergence', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Total Loss (Log Scale)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 2. Physics Loss Comparison
    plt.subplot(1, 3, 2)
    for name, data in results.items():
        plt.plot(epochs_range, data['history']['physics'], label=name, 
                 color=data['color'], linestyle=data['ls'], linewidth=2)
    plt.yscale('log')
    plt.title('Physics Loss Convergence', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Physics Residual Loss', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 3. Boundary Loss Comparison
    plt.subplot(1, 3, 3)
    for name, data in results.items():
        plt.plot(epochs_range, data['history']['bc'], label=name, 
                 color=data['color'], linestyle=data['ls'], linewidth=2)
    plt.yscale('log')
    plt.title('Boundary Condition Loss Convergence', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('BC Loss', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    save_path = 'images/hyperparam_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot successfully saved to '{save_path}'")
    plt.show()


if __name__ == '__main__':
    main()