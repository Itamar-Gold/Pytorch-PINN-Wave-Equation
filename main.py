import torch.optim as optim
import torch
from train import PINN as PINN
from train import train_model as train_model
import calculation_functions as cf
import test

train = True
save = True  # train the model and save the trained state of the model
load = False  # load the trained state of the model for use
prepare_animation = True  # prepare images for animation

model = PINN(2, 2, 32, 4)

if load:
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    model.load_state_dict(torch.load('model/model_weights.pth'))
    checkpoint = torch.load('model/model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if train:
    x, t = cf.data_sampler(100, [-1, 1], [0, 4])
    # Setup model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    inputs = torch.cat([x_tensor, t_tensor], dim=1)
    iters, losses = train_model(model, optimizer, inputs, x, t, prepare_animation)

    if save:
        torch.save(model.state_dict(), 'model/model_weights.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model/model_checkpoint.pth')

# test.show_slices(model, [-1, 1], [1.5, 3.5], 100)
