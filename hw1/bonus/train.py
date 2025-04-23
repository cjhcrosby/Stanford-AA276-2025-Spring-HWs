from argparse import ArgumentParser

import torch
import numpy as np
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.training.utils import current_git_hash

import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.multiprocessing.set_sharing_strategy('file_system')

resume_from_checkpoint = None
if os.path.exists('outputs/checkpoints2'):
    checkpoints = [f for f in os.listdir('outputs/checkpoint2') if f.endswith('.ckpt')]
    if checkpoints:
        resume_from_checkpoint = os.path.join('outputs/checkpoints2', sorted(checkpoints)[-1])
        print(f"Resuming from {resume_from_checkpoint}")

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [0, 0],
        [0.05, 0.05],
        [-0.05, -0.05],
        [0.05, -0.05],
        [-0.05, 0.05],
        [0.1, 0],
        [-0.1, 0],
        [0, 0.5],
        [0, -0.5]

    ]
)

simulation_dt = 0.01

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
args.gpus = 1

# Define the scenarios
nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
scenarios = [nominal_params]

# create system dynamics
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from part4 import state_limits
from part4 import control_limits
from part4 import safe_mask
from part4 import failure_mask
from part4 import f
from part4 import g

from neural_clbf.systems import inverted_pendulum

class CustomInvertedPendulum(inverted_pendulum.InvertedPendulum):
    def __init__(self, nominal_params, dt=0.01, controller_dt=None, scenarios=None):
        super().__init__(nominal_params, dt, controller_dt, scenarios)
    
    @property
    def state_limits(self):
        return state_limits()
    
    @property
    def control_limits(self):
        return control_limits()
    
    def safe_mask(self, x):
        return safe_mask(x)
    
    def _f(self, x, params):
        f_res = f(x)
        batch_size = x.shape[0]
        result = torch.zeros((batch_size, self.n_dims, 1), device=x.device)  
        result[:, 0, 0] = f_res[:, 0].clone()
        result[:, 1, 0] = f_res[:, 1].clone()
        return result
    
    def _g(self, x, params):
        g_res = g(x)
        batch_size = x.shape[0]
        result = torch.zeros((batch_size, self.n_dims, 1), device=x.device)  
        result[:, 0, 0] = g_res[:, 0].clone()
        result[:, 1, 0] = g_res[:, 1].clone()
        return result

dynamics_model = CustomInvertedPendulum(
    nominal_params,
    dt=simulation_dt,
    controller_dt=controller_period,
    scenarios=scenarios
)

initial_conditions = [
    (-np.pi, np.pi),
    (-8.0, 8.0)
]
data_module = EpisodicDataModule(
    dynamics_model,
    initial_conditions,
    trajectories_per_episode=50,
    trajectory_length=20,
    fixed_samples=100000,
    max_points=10000000,
    val_split=0.01,
    batch_size=1024,
    quotas={"safe": 0.7, "unsafe": 0.2, "goal": 0.1},
)

experiment_suite = ExperimentSuite([])

# Initialize the controller
cbf_controller = NeuralCBFController(
    dynamics_model,
    scenarios,
    data_module,
    experiment_suite=experiment_suite,
    cbf_hidden_layers=3,
    cbf_hidden_size=256, # lower size
    cbf_lambda=0.3,
    cbf_relaxation_penalty=1e4,
    controller_period=controller_period,
    primal_learning_rate=1e-2,
    scale_parameter=10.0, 
    learn_shape_epochs=5,
    use_relu=True,
    disable_gurobi=True,
)

# Initialize the logger and trainer
tb_logger = pl_loggers.TensorBoardLogger(
    'outputs',
    name='',
)

os.makedirs('outputs/checkpoint2', exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='Total loss / val',
    dirpath='outputs/checkpoints2',
    filename='cbf-{epoch:02d}-{val_total_loss:.2f}',  
    save_top_k=3,
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor="Total loss / val",
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode="min"
)

trainer = pl.Trainer.from_argparse_args(
    args,
    logger=tb_logger,
    reload_dataloaders_every_epoch=True,
    max_epochs=51,
    callbacks=[checkpoint_callback, early_stop_callback],
    resume_from_checkpoint=resume_from_checkpoint,
)

# Train
torch.autograd.set_detect_anomaly(True)
trainer.fit(cbf_controller)

if os.path.exists('outputs/checkpoints2'):
    print("Checkpoint files found:", os.listdir('outputs/checkpoints2'))
else:
    print("No checkpoint directory found!")

# Shutdown the VM after training completes
import os
print("Training complete. Shutting down the VM in 1 minute...")
os.system("sudo shutdown -h +1")  #