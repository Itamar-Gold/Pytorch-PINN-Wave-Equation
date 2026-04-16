# Physics-Informed Neural Network (PINN) for Wave Equation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

## About
In this mini-project, I practiced the design of a Physics-Informed Neural Network (PINN) that can solve a wave equation. The equation is a Partial Differential Equation (PDE) that arises to solve physical problems such as wave propagation. The PDE is: $∇^{2}h = -c^{2}h_{tt}$, where $∇^{2}$ is the Laplace operator, $c=1$, and $h(x,t)$ is the complex wavefield with real and imaginary components denoted by $h(x,t) = u(x,t) + iv(x,t)$.
The PINN is trained on the domain $x∈[−1,1]$ and $t ∈ [0, 4]$, for arbitrary initial conditions $(u_0(x, t), v_0(x, t))$.

This project implements advanced PINN methodologies including dynamic Latin Hypercube Sampling (LHS), and periodic batched sampling to prevent overfitting on collocation points.

## 📑 Table of Contents
* [About the Physics](#-about-the-physics)
* [Mathematical Formulation](#-mathematical-formulation)
* [Installation & Setup](#-installation--setup)
* [Usage Guide](#-usage-guide)
* [Results & Visualizations](#-results--visualizations)

---

## 🔬 About the Physics
This project trains a neural network to act as a surrogate solver for a Partial Differential Equation (PDE) that models physical wave propagation. 

The core PDE solved here is the 1D Wave Equation:
$$∇^{2}h = -c^{2}h_{tt}$$

Where:
* $∇^{2}$ is the Laplace operator.
* $c = 1$ is the wave propagation speed.
* $h(x,t)$ is the complex wavefield with real and imaginary components denoted by $h(x,t) = u(x,t) + iv(x,t)$.

The network learns to approximate this continuous function across the spatial domain $x \in [-1, 1]$ and temporal domain $t \in [0, 4]$.

## 🧮 Mathematical Formulation
The network is trained to satisfy arbitrary initial conditions $(u_0(x, t), v_0(x, t))$ and strict Dirichlet boundary conditions:

$$\begin{aligned}
h(-1, t) & =0 \\
h(1, t) & =0 \\
h(x, 0) & =u(x) \\
\left.\frac{\partial h(x, t)}{\partial t}\right|_{t=0} & =v(x)
\end{aligned}$$

### Analytical Ground Truth
To evaluate the network's accuracy, the loss gradients are compared against the exact analytical solution calculated via the variable separation method:

$$u(x, t) = \cos\left(\frac{\pi t}{4}\right) \sin\left(\frac{\pi (x + 1)}{2}\right)$$
$$v(x, t) = \sin\left(\frac{\pi t}{4}\right) \sin\left(\frac{\pi (x + 1)}{2}\right)$$
$$h(x, t) = u(x, t) + iv(x, t)$$

The network outputs predictions for $u(x, t)$ and $v(x, t)$ simultaneously via a multi-head architecture, which are then combined to form $|h(x, t)|$.

---

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/PINN-Wave-Equation.git
   cd PINN-Wave-Equation
   ```

2. **Install dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

*Note: The framework will automatically detect if a CUDA-enabled GPU is available and configure device allocation accordingly.*

---

## 💻 Usage Guide

The project utilizes `argparse` for clean and modular command-line configuration. All necessary directories (`/images`, `/model`) are generated automatically at runtime.

### Standard Training
To train the PINN using default hyperparameters and save the final model weights:
```bash
python main.py --train --save
```

### Evaluation Mode
To load a pre-trained model and evaluate it against the 2,500-point testing grid without retraining:
```bash
python main.py --load
```

### Animation Generation
To capture periodic frames of the 3D surface mapping during the training process:
```bash
python main.py --train --prep_anim
```
Once training finishes, stitch the frames together into a GIF:
```bash
# Animates a specific output
python create_animation.py --target "v(x,t)"

# Animates u, v, and h simultaneously
python create_animation.py --all
```

### Advanced Hyperparameter Configuration
You can fine-tune the architecture, learning rate, learning rate decay, loss penalty weights, and batch sizes directly from the CLI:
```bash
python main.py --train --epochs 2000 --lr 0.005 --hidden 64 --layers 6 --n_phys 1000 --resample_freq 50
```
Run `python main.py --help` for a full list of available parameters.

---

## 📊 Results & Visualizations

After training 2000 epochs on an NVIDIA GeForce RTX 4070 using dynamic LHS batched sampling and a decaying Adam optimizer, the PINN captures the wave propagation flawlessly.

### 3D Surface Predictions
**Real component $u(x,t)$:**
<img width="1400" height="700" alt="u_x_t_animation" src="https://github.com/user-attachments/assets/35040b48-1a82-44a8-ab22-cf7f356edfd1" />

**Imaginary component $v(x,t)$:**
<img width="1400" height="700" alt="v_x_t_animation" src="https://github.com/user-attachments/assets/115cb513-0de2-4e45-84fa-1625dad9f588" />

**Magnitude of complex wavefield $|h(x,t)|$:**
<img width="1400" height="700" alt="h_x_t_animation" src="https://github.com/user-attachments/assets/af312337-4f90-49b0-8992-758ba1ea2e60" />

### 2D Temporal Slices
Slicing the spatio-temporal domain to visualize the wave at specific time steps:
<img width="1000" height="400" alt="solution_for_t_ 0p5_2p3" src="https://github.com/user-attachments/assets/b1c1d7b0-d3b6-49eb-9c2e-a19a7e9819a5" />

### Network Accuracy
Calculated relative to the exact analytical solution across 2,500 randomly distributed test points:

* **$L_2$ Relative Error $E_{u(x,t)}$** = `2.86%`
* **$L_2$ Relative Error $E_{v(x,t)}$** = `1.67%`
