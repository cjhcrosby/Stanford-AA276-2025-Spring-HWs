import torch

"""
Helper class for querying cbf.ckpt.
neuralcbf = NeuralCBF()
h_values = neuralcbf.h_values(x)
h_gradients = neuralcbf.h_gradients(x)
"""
class NeuralCBF:
    def __init__(self, ckpt_path='../hw2/outputs/cbf.ckpt'):
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
    
# create a neuralcbf object
neuralcbf = NeuralCBF()
# define state bounds
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
batch_size = 10000
x = torch.rand(batch_size, 13)*(upper-lower)+lower
# get the h values and gradients
h_values = neuralcbf.values(x)
h_gradients = neuralcbf.gradients(x)
# print the shapes of the h values and gradients
print(f'h values shape: {h_values.shape}')
print(f'h gradients shape: {h_gradients.shape}')
# print h values above 0
print(f'h values above 0: {h_values[h_values > 0].shape[0]}')
# print h values below 0 divided by total samples
print(f'h values below 0: {h_values[h_values < 0].shape[0] / batch_size}')