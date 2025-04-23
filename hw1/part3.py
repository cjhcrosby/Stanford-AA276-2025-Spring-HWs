"""
AA 276 Homework 1 | Coding Portion | Part 3 of 3


OVERVIEW

In this file, you will implement functions for 
visualizing your learned CBF from Part 1 and evaluating
the accuracy of the learned CBF and corresponding CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 and Part 2 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, make sure that (in your VM) there is a CBF model checkpoint
saved at `outputs/cbf.ckpt`. Then, run `python scripts/plot.py`.
Submit the false safety rate reported in the terminal and the plot that is
saved to `outputs/plot.png`.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""


import torch
import pdb

def plot_h(fig, ax, px, py, slice, h_fn):
    """
    Plot a 2D visualization of the CBF values across the grid defined by
    px and py for the provided state slice onto the provided matplotlib figure and axes.
    Note: px/py (x/y position state variable) defines the state grid along the
        x-axis/y-axis of the plot.
    Note: We will add plot titles/labels for you; you just need to add the
        colormap, its corresponding colorbar, and the zero level set contour.
        
    args:
        fig: matplotlib Figure
        ax: matplotlib Axes
        px: torch float32 tensor with shape [npx]
        py: torch float32 tensor with shape [npy]
        slice: torch float32 tensor with shape [13] (first 2 elements can be ignored)
        h_fn: Callable h=h_fn(x)
            h_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size]
    """
    # Use indexing='ij' for newer PyTorch versions (or leave as default for older)
    PX, PY = torch.meshgrid(px, py)
    X = torch.zeros((len(px), len(py), 13))
    X[..., 0] = PX
    X[..., 1] = PY
    X[..., 2:] = slice[2:]

    h_values = h_fn(X.reshape(-1, 13)).reshape(len(px), len(py)) # reshape the mesh
        
    PX_np = PX.numpy()
    PY_np = PY.numpy()
    h_values_np = h_values.detach().cpu().numpy()
    
    v_abs_max = max(abs(h_values_np.min()), abs(h_values_np.max()))
    pcm = ax.pcolormesh(PX_np, PY_np, h_values_np, 
                        cmap='seismic_r', 
                        shading='auto',
                        vmin=-v_abs_max, vmax=v_abs_max)
    
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('CBF Value h(x)')
    
    levels = [0]
    if h_values_np.min() < 0 and h_values_np.max() > 0:
        ax.contour(PX_np, PY_np, h_values_np, levels=levels, 
                   colors='k', linewidths=2)
    
    # Force plot updates
    fig.canvas.draw()


from part1 import safe_mask, failure_mask
from part2 import roll_out, u_qp


def plot_and_eval_xts(fig, ax, x0, u_ref_fn, h_fn, dhdx_fn, gamma, lmbda, nt, dt):
    """
    First, compute the state trajectories xts starting from initial states x0 under the CBF-QP
    controller given by the reference controller u_ref_fn, the CBF h_fn, the CBF gradient dhdx_fn, and
    parameters gamma and lmbda for nt Euler steps with time step dt.
    Hint: we have imported roll_out(.) and u_qp(.) from Part 2 for you to use.
        
    Next, plot the state trajectories xts projected onto the 2D position space on the provided matplotlib figure and axes.
        Note: We will add plot titles/labels for you; you just need to add the trajectories.

    Finally, return the false_safety_rate. Specifically, of the initial states x0 that are in the safe set,
        what proportion result in trajectories that actually violate safety?
    Hint: we have imported safe_mask(x) and failure_mask(x) from Part 1 for you to use.

    args:
        fig: matplotlib Figure
        ax: matplotlib Axes
        x0: torch float32 tensor with shape [batch_size, 13]
        u_ref_fn: Callable u_ref=u_ref_fn(x)
            u_ref_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 4]
        h_fn: Callable h=h_fn(x)
            h_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size]
        dhdx_fn: Callable dhdx=dhdx_fn(x)
            dhdx_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 13]
        gamma: float
        lmbda: float
        nt: int
        dt: float

    returns:
        false_safety_rate: float
    """
    # here is some starting code that defines the controller you should be using
    def u_fn(x):
        return u_qp(x, h_fn(x), dhdx_fn(x), u_ref_fn(x), gamma, lmbda)
    # first, you should compute state trajectories xts using roll_out(.)
    xts = roll_out(x0, u_fn, nt, dt) # compute trajectpries

    xts_np = xts.detach().cpu().numpy() # make a numpy copy
    batch_size = x0.shape[0]
    for i in range(batch_size): # loop thru batches
        positions = xts_np[i, :, :2]
        safe0 = h_fn(x0[i:i+1]).item()
        color = 'blue' if safe0 >= 0 else 'red'
        ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=0.5)
        ax.scatter(positions[0, 0], positions[0, 1], color=color, marker='o')
    initially_safe = safe_mask(x0)
    failures = torch.zeros_like(initially_safe)

    for i in range(batch_size):
        if initially_safe[i]:
            for t in range(xts.shape[1]):
                state_t = xts[i,t].unsqueeze(0)
                if failure_mask(state_t).any():
                    failures[i] = True
                    break
    false_safety_rate = torch.sum(failures) / torch.sum(initially_safe)

    return false_safety_rate.item()
