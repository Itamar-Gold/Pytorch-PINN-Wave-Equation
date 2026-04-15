import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Workaround to fix imports if there's any circular dependency issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import PINN, train_model, create_dataset
import calculation_functions as cf
import eval_model

# Setup device configuration globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    """
    Parses command line arguments to configure the training setup dynamically,
    removing the hardcoded global variables and allowing flexibility.
    """
    parser = argparse.ArgumentParser(description="Physics-Informed Neural Network (PINN) for the Wave Equation")
    
    # Action modes
    parser.add_argument('--train', action='store_true', help='Enable training mode.')
    parser.add_argument('--save', action='store_true', help='Save the trained model and optimizer state.')
    parser.add_argument('--load', action='store_true', help='Load a pre-trained model state for evaluation/continue training.')
    parser.add_argument('--prep_anim', action='store_true', help='Prepare images during training to render an animation.')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=2000, help='Total number of training iterations.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
    parser.add_argument('--decay', type=float, default=0.999, help='Exponential learning rate decay factor.')
    
    # Model Architecture
    parser.add_argument('--layers', type=int, default=8, help='Number of layers in the network.')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units per layer.')
    
    # Collocation point sampling
    parser.add_argument('--n_ic', type=int, default=50, help='Number of Initial Condition collocation points per batch.')
    parser.add_argument('--n_bc', type=int, default=50, help='Number of Boundary Condition collocation points per batch.')
    parser.add_argument('--n_phys', type=int, default=250, help='Number of Physics collocation points per batch (interior domain).')
    parser.add_argument('--n_multiplier', type=int, default=10, help='Multiplier for creating the large pre-computed dataset.')
    
    # Loss Weights
    parser.add_argument('--w_ic', type=float, default=0.01, help='Weight for Initial Condition loss.')
    parser.add_argument('--w_bc', type=float, default=0.01, help='Weight for Boundary Condition loss.')
    parser.add_argument('--w_phys', type=float, default=0.001, help='Weight for Physics residual loss.')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"Using compute device: {device}")
    
    # Create required directories
    os.makedirs('images/u(x,t)', exist_ok=True)
    os.makedirs('images/v(x,t)', exist_ok=True)
    os.makedirs('images/h(x,t)', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # Initialize the model architecture based on arguments
    # Input dim: 2 (x, t), Output dim: 2 (u, v)
    model = PINN(n_input=2, n_output=2, n_hidden=args.hidden, n_layers=args.layers).to(device)
    
    # Setup optimizer (Adam is standard for PINNs, L-BFGS often used later)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    # Setup learning rate scheduler to handle stiff gradient issues common in PINNs
    scheduler = ExponentialLR(optimizer, gamma=args.decay)

    # If nothing is specified, assume user just wants to evaluate an existing model
    if not args.train and not args.load:
        print("No actions specified. Running default: --load")
        args.load = True

    # Load pre-trained state if requested
    if args.load:
        print("Loading pre-trained model weights...")
        try:
            model.load_state_dict(torch.load('model/model_weights.pth', map_location=device))
            checkpoint = torch.load('model/model_checkpoint.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully loaded model state.")
        except FileNotFoundError:
            print("Error: Could not find pre-trained weights in the 'model' directory. Please train first using --train and --save.")
            return
        except Exception as e:
             print(f"Error loading model: {e}")
             return

    # Train mode
    if args.train:
        print("Starting training process...")
        # Sample base grid data points for evaluation loss (Supervised component)
        x, t = cf.data_sampler(100, [-1, 1], [0, 4])
        
        # Remove requires_grad=True here because supervised loss components do not need gradients of inputs
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
        t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
        inputs = torch.cat([x_tensor, t_tensor], dim=1)
        
        # Define base batch sizes and weights
        batch_sizes = (args.n_ic, args.n_bc, args.n_phys)
        weights = (args.w_ic, args.w_bc, args.w_phys)
        
        # Generate the large overarching dataset for batch sampling
        print(f"Creating large pre-computed dataset for batching (Multiplier: {args.n_multiplier}x)...")
        dataset = create_dataset(n_points_multiplier=args.n_multiplier, base_n_points=batch_sizes)
        
        _, _ = train_model(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler,
            inputs=inputs, 
            epochs=args.epochs, 
            weights=weights,
            batch_sizes=batch_sizes,
            dataset=dataset,
            animation_prep=args.prep_anim
        )

        # Save trained state if requested
        if args.save:
            print("Saving trained model...")
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), 'model/model_weights.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model/model_checkpoint.pth')
            print("Model saved successfully.")

    # Post-training evaluation
    print("Evaluating model performance...")
    # Move model to CPU for testing/plotting routines if they rely heavily on NumPy
    model.cpu()
    model.eval()
    
    x_test, t_test = cf.data_sampler(2500, [-1, 1], [0, 4])
    
    # Show slices at specific times using eval_model (formerly test.py) to prevent standard library conflicts
    eval_model.show_slices(model, [-1, 1], [0.5, 2.3], 100)
    
    # Calculate error norms
    u_error, v_error = eval_model.error_l2norm(model, x_test, t_test)
    print(f"L2 Relative Error -> U: {u_error}, V: {v_error}")


if __name__ == "__main__":
    main()