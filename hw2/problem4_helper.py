"""
Helper functions to query the learned values and gradients of vf.ckpt, cbf.ckpt.
NOTE: To use NeuralCBF defined below, you need to have your solution files from Homework 1
copied to this directory. Alternatively, write your script that uses NeuralCBF
in your ../hw1/ folder from Homework 1 (a copy of this file should already be there).
Make sure your hw1 venv is activated to use NeuralCBF.
NOTE: Due to package version incompatibilities between the two neural libraries,
you will probably need to use separate scripts and venvs.
"""

import torch

def get_device():
    """Get optimal available device (MPS for Mac, CUDA if available, otherwise CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
# Global device variable
device = get_device()
print(f"Using device in losses.py: {device}")
"""
Helper class for querying vf.ckpt.
neuralvf = NeuralVF()
values = neuralvf.values(x)
gradients = neuralvf.gradients(x)
"""
class NeuralVF:
    def __init__(self, ckpt_path='outputs/vf.ckpt'):
        import os
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

        from libraries.DeepReach_MPC.utils import modules
        from libraries.DeepReach_MPC.dynamics.dynamics import Quadrotor

        dynamics = Quadrotor(collisionR=0.5, collective_thrust_max=20, set_mode='avoid')
        model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type='sine', mode='mlp',
                                    final_layer_factor=1., hidden_features=512, num_hidden_layers=3, 
                                    periodic_transform_fn=dynamics.periodic_transform_fn)
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path)['model'])

        self.dynamics = dynamics
        self.model = model

    def values(self, x):
        """
        args:
            x: torch tensor with shape      [batch_size, 13]
        returns:
            values: torch tensor with shape [batch_size]
        """
        coords = torch.concatenate((torch.ones((len(x), 1)), x), dim=1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            model_results = self.model({'coords': model_input.to(device)})
        values = self.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].detach().squeeze(dim=-1))
        return values.cpu()
    
    def gradients(self, x):
        """
        args:
            x: torch tensor with shape         [batch_size, 13]
        returns:
            gradients: torch tensor with shape [batch_size, 13]
        """
        coords = torch.concatenate((torch.ones((len(x), 1)), x), dim=1)
        model_input = self.dynamics.coord_to_input(coords)
        model_results = self.model({'coords': model_input.to(device)})
        gradients = self.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))[:, 1:]
        return gradients.cpu()
    
"""
Helper class for querying cbf.ckpt.
neuralcbf = NeuralCBF()
h_values = neuralcbf.h_values(x)
h_gradients = neuralcbf.h_gradients(x)
"""
class NeuralCBF:
    def __init__(self, ckpt_path='outputs/cbf.ckpt'):
        try:
            from neural_clbf.controllers import NeuralCBFController
        except Exception as e:
            print(str(e))
            print('MAKE SURE YOU HAVE THE VENV FROM HW1 ACTIVATED')
        try:
            self.model = NeuralCBFController.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(str(e))
            print('MAKE SURE YOUR FILES FROM HOMEWORK 1 ARE IN THE SAME DIRECTORY AS THIS FILE')

    def values(self, x):
        """
        args:
            x: torch tensor with shape    [batch_size, 13]
        
        returns:
            h(x): torch tensor with shape [batch_size]
        """
        return -self.model.V_with_jacobian(x)[0]
    
    def gradients(self, x):
        """
        args:
            x: torch tensor with shape       [batch_size, 13]

        returns:
            dhdx(x): torch tensor with shape [batch_size, 13]
        """
        return -self.model.V_with_jacobian(x)[1].squeeze(1)
        
neuralvf = NeuralVF()

p_u = 3.0
p_l = -3.0
q_u = 1.0
q_l = -1.0
v_u = 5.0
v_l = -5.0
w_u = 5.0
w_l = -5.0
upper = torch.tensor([p_u,p_u,p_u,q_u,q_u,q_u,q_u,v_u,v_u,v_u,w_u,w_u,w_u])
lower = torch.tensor([p_l,p_l,p_l,q_l,q_l,q_l,q_l,v_l,v_l,v_l,w_l,w_l,w_l])

# create a large batch of random states to randomly sample from
h_values = []
batch_size = 10000
for i in range(10):
    x = torch.rand(batch_size, 13)*(upper-lower)+lower
    h_values.append(neuralvf.values(x))
        
h_values = torch.cat(h_values, dim=0)
# get the h values and gradients
h_gradients = neuralvf.gradients(x)
# print the shapes of the h values and gradients
print(f'h values shape: {h_values.shape}')
# print h values above 0
print(f'h values above 0: {h_values[h_values >= 0].shape[0]}')
# print h values above 0 divided by total samples
print(f'h values above 0: {h_values[h_values >= 0].shape[0] / h_values.shape[0]}')
