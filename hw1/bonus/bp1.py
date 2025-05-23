import torch
import os
import sys
import numpy as np
import cvxpy as cp
from matplotlib import colormaps
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import pdb
from neural_clbf.controllers import NeuralCBFController
from neural_clbf.systems import inverted_pendulum

# Get absolute path to the directory containing part4.py
hw1_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(hw1_dir)  # Add to Python's path

# Now try importing
try:
    from part4 import state_limits, control_limits, safe_mask
    from part4 import f as part4_f, g as part4_g
    print("Successfully imported part4 module")
except ImportError:
    print(f"Looking for part4.py in: {hw1_dir}")
    print(f"Files in directory: {os.listdir(hw1_dir)}")
    raise


class CustomInvertedPendulum(inverted_pendulum.InvertedPendulum):
    def __init__(self, nominal_params, dt=0.01, controller_dt=None, scenarios=None):
        super().__init__(nominal_params, dt, controller_dt, scenarios)
    
    # Override methods with your part4 implementations
    @property
    def state_limits(self):
        return state_limits()
    
    @property
    def control_limits(self):
        return control_limits()
    
    def safe_mask(self, x):
        return safe_mask(x)
    
    def _f(self, x, params):
        # Adapt your f function to return the right shape
        f_res = part4_f(x)  # Use the imported function
        batch_size = x.shape[0]
        result = torch.zeros((batch_size, self.n_dims, 1), device=x.device)
        result[:, 0, 0] = f_res[:, 0]
        result[:, 1, 0] = f_res[:, 1]
        return result
    
    def _g(self, x, params):
        g_res = part4_g(x)  # Use the imported function
        batch_size = x.shape[0]
        result = torch.zeros((batch_size, self.n_dims, 1), device=x.device)
        result[:, 0, 0] = g_res[:, 0]
        result[:, 1, 0] = g_res[:, 1]
        return result

# breakpoint()

a = 1/10
b = 1/5
m = 2.0
l = 1.0
g = 10.0

def h(x):
    """
    Function to compute the value of h(x) = 1 - 1/(a^2) x[0]^2 - 1/(b^2) x[1]^2)
    """
    return 1 - 1/(a**2) * x[0]**2 - 1/(b**2) * x[1]**2
def dh(x,u):
    """
    Function to compute the value of dh(x) = -\dot{\theta} \left(
    \frac{2}{a^{2}} \theta + \frac{2g}{lb^{2}} \sin{ \theta} + \frac{2}{ml^{2}b^{2}} \mathbf{u}
    \right)
    """
    theta = x[0]
    dtheta = x[1]
    return -dtheta * (2/a**2 * theta + 2*g/(l*b**2) * np.sin(theta) + 2/(m*l**2*b**2) * u)

def f(x,u):
    """
    Function to compute the dynamics of the system,
    \left[\begin{array}{l}
    \dot{\theta} \\  \ddot{\theta}
    \end{array}\right] = \left[\begin{array}{l}
    \dot{\theta} \\ 
    \frac{g}{l} \sin{\theta} + \frac{1}{ml^{2}}u
    \end{array}\right]
    """
    theta = x[0]
    dtheta = x[1]
    a1 = dtheta
    a2 = np.float64(((g/l) * np.sin(theta) + (1/(m*l**2)) * u)[0])
    return np.array([a1, a2])

def u_nom(x,t):
    """
    Function to compute the nominal control input,
    u_\text{nom}(t) = \left\{\begin{array}{l}
    3, & t \in [0,1)\\
    -3, & t \in [1,2)\\
    3, & t \in [2,3)\\
    ml^{2}\left( -\frac{g}{l}\sin{\theta} - \left[\begin{array}{l}
    1.5 \\ 1.5
    \end{array}\right] \mathbf{x} \right) & \text{else}
    \end{array}\right.
    """
    if t < 1 and t >= 0:
        return 3.0
    elif t < 2 and t >= 1:
        return -3.0
    elif t < 3 and t >= 2:
        return 3.0
    else:
        return m*l**2 * (-g/l * np.sin(x[0]) - np.array([1.5, 1.5]) @ x)


def plot_trajectory(x_vals, tspan, cmap):
    theta = np.array(x_vals)[:,0]
    dtheta = np.array(x_vals)[:,1]
    # cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(tspan.min(), tspan.max())
    points = np.array([theta, dtheta]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(tspan)
    lc.set_linewidth(2)
    return lc


def main2_3():
    thetas = np.linspace(-a,a, 100)
    dthetas = np.linspace(-b,b, 100)
    theta, dtheta = np.meshgrid(thetas, dthetas)

    # Plot h(x) over the 2D state space with the zero-level set of h(x) and the failure set
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(theta, dtheta, h([theta, dtheta]), levels=30, cmap='managua',alpha=0.5) # Levels and cmap added
    plt.colorbar(contour, label='h(x)') 
    plt.contour(theta, dtheta, h([theta, dtheta]), levels=[0], colors='blue', linestyles='-', label='Zero-Level Set of h(x)')
    plt.plot(0, 0, color='blue', label='Zero-Level Set of h(x)') # Plot the zero-level set of h(x)
    xl = -0.5
    xh = 0.5
    yl = -0.5
    yh = 0.5
    plt.axvspan(-0.3, -100, color='red', alpha=0.3, label='Failure Set')
    plt.axvspan(0.3, 100, color='red', alpha=0.3)


    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\dot{\theta}$')
    plt.title('h(x) and Failure Set')
    plt.grid()
    plt.legend()
    plt.xlim(xl, xh)
    plt.ylim(yl, yh)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.show()

def plot_neural_vs_analytical_cbf():
    checkpoint_path = 'outputs/checkpoints/cbf-epoch=05-val_total_loss=0.00.ckpt'
    
    # Load model
    model = NeuralCBFController.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Create test points
    thetas = np.linspace(-a, a, 100)
    dthetas = np.linspace(-b, b, 100)
    theta, dtheta = np.meshgrid(thetas, dthetas)
    
    grid_points = np.stack([theta.flatten(), dtheta.flatten()], axis=1)
    x_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # Debug - print available attributes
    print("Available model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
    
    # Extract the CBF value correctly - use the appropriate method from the model
    with torch.no_grad():
        # Try these options based on the NeuralCBFController API:
        try:
            # Option 1: Use V method (used for both CLF and CBF in this framework)
            neural_values = model.V(x_tensor).cpu().numpy().reshape(theta.shape)
        except AttributeError:
            try:
                # Option 2: Use cbf_net directly
                neural_values = model.cbf_net(x_tensor).cpu().numpy().reshape(theta.shape)
            except AttributeError:
                # Option 3: Try other common naming patterns
                neural_values = model.h(x_tensor).cpu().numpy().reshape(theta.shape)
    
    # Rest of your visualization code...
    analytical_values = h([theta, dtheta])
    dh_values = dh([theta, dtheta], u_nom([theta, dtheta], 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
    # Analytical CBF
    c1 = ax1.contourf(theta, dtheta, analytical_values, levels=30, cmap='viridis')
    ax1.contour(theta, dtheta, analytical_values, levels=[0], colors='white', linewidths=2)
    fig.colorbar(c1, ax=ax1, label='Analytical CBF Value')
    ax1.set_title('Analytical CBF')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\dot{\theta}$')
    ax1.grid(True)
    
    # Neural CBF
    c2 = ax2.contourf(theta, dtheta, neural_values, levels=30, cmap='plasma')
    ax2.contour(theta, dtheta, neural_values, levels=[0], colors='white', linewidths=2)
    fig.colorbar(c2, ax=ax2, label='Neural CBF Value')
    ax2.set_title('Learned Neural CBF')
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\dot{\theta}$')
    ax2.grid(True)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.axvspan(-0.3, -100, color='red', alpha=0.2)
        ax.axvspan(0.3, 100, color='red', alpha=0.2)
    
    plt.suptitle('Comparison of Analytical vs Neural CBF', fontsize=16)
    plt.tight_layout()
    plt.savefig("cbf_comparison.png", dpi=300)
    plt.show()
    
    # Also create overlay plot
    plt.figure(figsize=(10, 8))
    plt.contour(theta, dtheta, analytical_values, levels=[0], colors='blue', 
                linewidths=3, linestyles='-', label='Analytical Zero-Level Set')
    plt.contour(theta, dtheta, neural_values, levels=[0], colors='red', 
                linewidths=3, linestyles='--', label='Neural Zero-Level Set')
    plt.axvspan(-0.3, -100, color='gray', alpha=0.3)
    plt.axvspan(0.3, 100, color='gray', alpha=0.3)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.grid(True)
    plt.title('Zero-Level Sets Comparison')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\dot{\theta}$')
    plt.legend()
    plt.savefig("zero_level_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # main2_3()  
    plot_neural_vs_analytical_cbf()