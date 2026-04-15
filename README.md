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
   git clone https://github.com/YourUsername/Pytorch-PINN-Wave-Equation.git
   cd Pytorch-PINN-Wave-Equation
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
python main.py --train --epochs 6000 --lr 0.005 --hidden 64 --layers 6 --n_phys 1000 --resample_freq 50
```
Run `python main.py --help` for a full list of available parameters.

---

## 📊 Results & Visualizations

After 4,000 epochs of training on an Nvidia GPU using dynamic LHS batched sampling and a decaying Adam optimizer, the PINN captures the wave propagation flawlessly.

### 3D Surface Predictions
**Real component $u(x,t)$:**
![u(x,t)](https://github.com/Itamar-Gold/Pytorch-PINN-Wave-Equation/assets/92544992/840f770c-0b69-4410-a433-f6678bc32410)

**Imaginary component $v(x,t)$:**
![v(x,t)](https://github.com/Itamar-Gold/Pytorch-PINN-Wave-Equation/assets/92544992/6bc6f056-737e-49b2-a69f-80709d2b8f10)

**Magnitude of complex wavefield $|h(x,t)|$:**
![h(x,t)](https://github.com/Itamar-Gold/Pytorch-PINN-Wave-Equation/assets/92544992/e037b8d9-5e6b-4293-81f6-092eaaa07b39)

### 2D Temporal Slices
Slicing the spatio-temporal domain to visualize the wave at specific time steps:

![solution for t =  0 5, 2 3](https://github.com/Itamar-Gold/Pytorch-PINN-Wave-Equation/assets/92544992/34713540-f20b-4750-a3b7-cc6e0017f29c)
![solution for t =  1 5, 3 5](https://github.com/Itamar-Gold/Pytorch-PINN-Wave-Equation/assets/92544992/ed09ff25-c69f-4c50-b84f-7e19c1e1d095)

### Network Accuracy
Calculated relative to the exact analytical solution across 2,500 randomly distributed test points:

* **$L_2$ Relative Error $E_{u(x,t)}$** = `2.86%`
* **$L_2$ Relative Error $E_{v(x,t)}$** = `1.67%`
