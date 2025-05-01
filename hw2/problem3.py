import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from scipy.interpolate import interpn
from matplotlib import collections as mcoll

import hj_reachability as hj
from hj_reachability import dynamics
from hj_reachability import sets
from tqdm import tqdm

u_bar = 3.0  # control bound
class InvertedPendulum(dynamics.ControlAndDisturbanceAffineDynamics):
  def __init__(self,
               m=2.,
               l=1.,
               g=10.,
               u_bar=u_bar):
    self.m = m
    self.l = l
    self.g = g
    control_mode = 'max'
    disturbance_mode = 'min'
    control_space = sets.Box(jnp.array([-u_bar]), jnp.array([u_bar]))
    disturbance_space = sets.Box(jnp.array([0.]), jnp.array([0.])) # No disturbance
    super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

  def open_loop_dynamics(self, state, time):
    theta, dtheta = state
    f = jnp.array([dtheta, (self.g / self.l) * jnp.sin(theta)])
    return f
  def control_jacobian(self, state, time):
    g = jnp.array([[0.], [1/(self.m * self.l ** 2)]])
    return g

  def disturbance_jacobian(self, state, time):
    return jnp.array([[0.], [0.]])
  

# Define the computation grid for numerically solving the PDE
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(np.array([-jnp.pi, -10.]), # lower bounds
                np.array([jnp.pi, 10.])),  # upper bounds
    (101, 101))

# Define the implicit function l(x) for the failure set
failure_values = 0.3 - jnp.abs(grid.states[..., 0])

# Solver settings
times = np.linspace(0, -5, 101, endpoint=True)
solver_settings = hj.SolverSettings.with_accuracy('very_high',
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

dynamics = InvertedPendulum()
print("Computing BRT...")
values = hj.solve(solver_settings, dynamics, grid, times, failure_values)
print("Done.")

grads_T = grid.grad_values(values[-1], solver_settings.upwind_scheme)
alpha2_T = grads_T[:, :, 1]



def create_interactive_plot():
    """Plot the value function and failure set with zero-level set contour and slider over time."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25) # pad bottom for slider
    
    t_init = 0
    ti_init = np.argwhere(np.isclose(times, t_init)).item() # initial time
    
    title = ax.set_title(f'$V(x, {t_init})$')
    ax.set_xlabel('$\\theta$ (rad)')
    ax.set_ylabel('$\\dot{\\theta}$ (rad/s)')
    
    pcm = ax.pcolormesh( 
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[ti_init].T,
        cmap='RdBu',
        vmin=-vbar, vmax=vbar
    ) 
    fig.colorbar(pcm, ax=ax)
    
    contour1 = ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[ti_init].T,
        levels=0,
        colors='k'
    ) # Value function contour
    contour2 = ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        failure_values.T,
        levels=0,
        colors='r'
    ) # Failure set contour
    
    # Time Slider
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    time_slider = Slider(
        ax=ax_time,
        label='Time',
        valmin=times[-1],  
        valmax=times[0],
        valinit=t_init,
    )
    
    # Update slider 
    def update(val):
        nonlocal contour1 # Remove previous contour

        t = time_slider.val
        ti = np.argmin(np.abs(times - t)) # closest time to slider value
        
        # Update title to include current time
        title.set_text(f'$V(x, {t:.2f})$')
        
        # Update pcolormesh
        pcm.set_array(values[ti].T.ravel())
        
        # Remove old contours and add new ones
        for coll in contour1.collections:
            coll.remove()
        new_contour1 = ax.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            values[ti].T,
            levels=0,
            colors='k'
        )
        contour1 = new_contour1
        fig.canvas.draw_idle()
    time_slider.on_changed(update)
    plt.show()

def save_values_gif(values, grid, times, save_path='outputs/values.gif'):
    """
    args:
        values: ndarray with shape [
                len(times),
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: hj.Grid
        times: ndarray with shape [len(times)]
    """
    vbar = 3
    fig, ax = plt.subplots()
    ax.set_title(f'$V(x, {times[0]:3.2f})$')
    ax.set_xlabel('$\\theta$ (rad)')
    ax.set_ylabel('$\\dot{\\theta}$ (rad/s)')
    value_plot = ax.pcolormesh(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[0].T,
        cmap='RdBu',
        vmin=-vbar,
        vmax=+vbar
    )
    plt.colorbar(value_plot, ax=ax)
    global value_contour
    value_contour = ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[0].T,
        levels=0,
        colors='k'
    )

    def update(i):
        ax.set_title(f'$V(x, {times[i]:3.2f})$')
        value_plot.set_array(values[i].T)
        global value_contour
        value_contour.remove()
        value_contour = ax.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            values[i].T,
            levels=0,
            colors='k'
        )
        return ax
    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=np.arange(len(times)),
        interval=int(1000*(-times[1]))
    )
    with tqdm(total=len(times)) as anim_pbar:
        anim.save(filename=save_path, writer='pillow', progress_callback=lambda i, n: anim_pbar.update(1))
    print(f'SAVED GIF TO: {save_path}')
    plt.close()

def get_volume(values, grid, times):
   """Get volume of the BRT using the super zero level set of the value function."""
#    breakpoint()
   super_zero = sum(sum(values[-1,:,:]>=0)).item()
   return super_zero
   
def plot_value_and_safe_set_boundary(values_converged, grid, ax):
    """
    args:
        values_converged: ndarray with shape [
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: hj.Grid,
        ax: matplotlib axes to plot on
    """
    values_converged_interpolator = RegularGridInterpolator(
        ([np.array(v) for v in grid.coordinate_vectors]),
        np.array(values_converged),
        bounds_error=False,
        fill_value=None
    )
    vbar=3
    vis_thetas = np.linspace(-0.5, +0.5, num=101, endpoint=True)
    vis_theta_dots = np.linspace(-1, +1, num=101, endpoint=True)
    vis_xs = np.stack((np.meshgrid(vis_thetas, vis_theta_dots, indexing='ij')), axis=2)
    vis_values_converged = values_converged_interpolator(vis_xs)
    ax.pcolormesh(
        vis_thetas,
        vis_theta_dots,
        vis_values_converged.T,
        cmap='RdBu',
        vmin=-vbar,
        vmax=vbar
    )
    ax.contour(
        vis_thetas,
        vis_theta_dots,
        vis_values_converged.T,
        levels=[0],
        colors='k'
    )
    
    contour2 = ax.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            failure_values.T,
            levels=0,
            colors='r'
            ) # Failure set contour
    

def optimal_safety_controller(x):
  """Implement the optimal safety controller of the form
    \mathbf{u}_\text{safe}(\mathbf{x},t)= \arg\max\limits_{u\in \mathcal{U}} \frac{\partial{V}(\mathbf{x},t)}{\partial{\mathbf{x}}} \cdot f(\mathbf{x},u)
  """
  alpha2 = interpn(
      ([np.array(v) for v in grid.coordinate_vectors]),
      np.array(alpha2_T),
      x,
      method='linear',
      bounds_error=False,
      fill_value=None
  )
  return np.sign(alpha2).item()*u_bar # bang-bang

T = 1.0
dt = 0.01 # change this to 0.001 for better control
def simulate(x0):
    nt = int(T / dt)
    xs = np.full((nt, 2), fill_value=np.nan)
    us = np.full((nt, 1), fill_value=np.nan)
    xs[0] = x0
    us[0] = optimal_safety_controller(x0)
    for i in range(1, nt):
        x = xs[i-1]
        u = optimal_safety_controller(x)
        us[i] = u
        xs[i] = x + dt*np.array([x[1], 12*u-9.8])
    return xs, us

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


def plot_optimal_safety(x0s, values_converged, grid):
    # breakpoint()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_value_and_safe_set_boundary(values_converged, grid, ax=plt.gca())
    xs = [None] * len(x0s)  # Pre-allocate list
    us = [None] * len(x0s)  # Pre-allocate list
    lc = [None] * len(x0s)  # Pre-allocate list
    cmaps = ['plasma', 'viridis'] #, 
    colors = ['r','b'] # 'r', 
    for i in range(len(x0s)):
        xs[i], us[i] = simulate(x0s[i])
        lc[i] = plot_trajectory(xs[i], np.linspace(0, T, len(xs[i])), cmap=cmaps[i])
        ax.plot(xs[i][0,0], xs[i][0,1], colors[i]+'o', markersize=10, label='Initial State')
        ax.plot(xs[i][-1, 0], xs[i][-1, 1], colors[i]+'x', markersize=10, label='Final State')
        ax.add_collection(lc[i])
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-1, 1])
    ax.legend()

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(x0s)):
        ax.plot(us[i][:,0], color=colors[i], label=f'Control for x0_{i+1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Control Input')
    ax.set_title('Control Input Over Time')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    
    plt.show()


   
if __name__ == "__main__":
    vbar = np.max(np.abs(values)) # max value of BRT
    
    # # # # 3.1 and 3.2
    # create_interactive_plot()
    # save_values_gif(values, grid, times, save_path='outputs/values.gif')

    # # # # 3.3
    # volume = get_volume(values, grid, times)
    # print(f'Volume of the BRT: {volume}, which is {100*volume/(grid.coordinate_vectors[0].shape[0]*grid.coordinate_vectors[1].shape[0])}% of the total volume of the state space.')
    
    # # # # 3.4
    x0_1 = np.array([-0.1, 0.4])
    x0_2 = np.array([-0.1, -0.3])
    x0s = np.array([x0_1, x0_2]) # will plot all onto one figure
    plot_optimal_safety(x0s, values[-1], grid)
    
    