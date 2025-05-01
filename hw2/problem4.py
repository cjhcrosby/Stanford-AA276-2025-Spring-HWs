import torch

"""For your convenience, we have provided f(x) from Homework 1."""
def f(x):
    """
    Return the control-independent part of the control-affine dynamics for the 13D quadrotor system.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        
    returns:
        f: torch float32 tensor with shape [batch_size, 13]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    f = torch.zeros_like(x)
    f[:, PXi] = VX
    f[:, PYi] = VY
    f[:, PZi] = VZ
    f[:, QWi] = -0.5*(WX*QX + WY*QY + WZ*QZ)
    f[:, QXi] =  0.5*(WX*QW + WZ*QY - WY*QZ)
    f[:, QYi] =  0.5*(WY*QW - WZ*QX + WX*QZ)
    f[:, QZi] =  0.5*(WZ*QW + WY*QX - WX*QY)
    f[:, VZi] = -9.8
    f[:, WXi] = -5 * WY * WZ / 9.0
    f[:, WYi] =  5 * WX * WZ / 9.0
    return f

"""For your convenience, we have provided g(x) from Homework 1."""
def g(x):
    """
    Return the control-dependent part of the control-affine dynamics for the 13D quadrotor system.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    g = torch.zeros((*x.shape, 4))
    g[:, VXi, 0] = 2 * (QW*QY + QX*QZ)
    g[:, VYi, 0] = 2 * (QY*QZ - QW*QX)
    g[:, VZi, 0] = (1 - 2*torch.pow(QX, 2) - 2*torch.pow(QY, 2))
    g[:, WXi:, 1:] = torch.eye(3)

    return g

"""IMPLEMENT THE FOLLOWING FUNCTIONS"""

def optimal_control(x, dVdx):
    """
    Compute the optimal safety controller for the 13D quadrotor system for
    states x and
    value function spatial gradients dVdx.

    args:
        x:     torch tensor with shape [batch_size, 13]
        dVdx:  torch tensor with shape [batch_size, 13]
    
    returns:
        u_opt: torch tensor with shape [batch_size, 4]
    """
    F_u = 20.0
    F_l = -20.0
    a_x_u = 8.0
    a_x_l = -8.0
    a_y_u = 8.0
    a_y_l = -8.0
    a_z_u = 4.0
    a_z_l = -4.0
    upper = torch.tensor([F_u, a_x_u, a_y_u, a_z_u])
    lower = torch.tensor([F_l, a_x_l, a_y_l, a_z_l])

    batch_size = x.shape[0]
    u_opt = torch.zeros((x.shape[0], 4))
    
    # compute alpha2 = dVdx^T*g(x) but with batch matrix multiplication
    alpha2 = torch.bmm(g(x).transpose(1,2), dVdx.unsqueeze(2)).squeeze(2)

    # use bang bang structure and torch.where from DeepReach
    u_opt[:, 0] = torch.where(alpha2[:,0] > 0 , upper[0], lower[0])
    u_opt[:, 1] = torch.where(alpha2[:,1] > 0 , upper[1], lower[1])
    u_opt[:, 2] = torch.where(alpha2[:,2] > 0 , upper[2], lower[2])
    u_opt[:, 3] = torch.where(alpha2[:,3] > 0 , upper[3], lower[3])

    return u_opt

def hamiltonian(x, dVdx):
    """
    Compute the Hamiltonian for the 13D quadrotor system for
    states x and
    value function spatial gradients dVdx.

    args:
        x:    torch tensor with shape [batch_size, 13]
        dVdx: torch tensor with shape [batch_size, 13]

    returns:
        ham:  torch tensor with shape [batch_size]
    """
    ham = torch.zeros(x.shape[0])
    # compute alpha1 = dVdx^T*f(x) but with batch matrix multiplication
    alpha1 = torch.bmm(f(x).unsqueeze(1), dVdx.unsqueeze(2)).squeeze(1).squeeze(1)
    # breakpoint()
    # compute alpha2 = dVdx^T*g(x) with batch matrix multiplication same as in optimal control
    alpha2 = torch.bmm(g(x).transpose(1,2), dVdx.unsqueeze(2)).squeeze(2)
    # breakpoint()
    u_opt = optimal_control(x, dVdx)
    
    return alpha1 + torch.sum(alpha2 * u_opt, dim=1)


def hji_vi_loss(x, l, V, dVdt, dVdx):
    """
    Compute the HJI-VI loss term for the
    states x,
    failure function values l,
    value function values V,
    value function temporal gradients dVdt, and
    value function spatial gradients dVdx.
    This corresponds to h2(.) in Equation (14)
    in the DeepReach paper: https://arxiv.org/pdf/2011.02082.
    NOTE: You should return a batch of losses, i.e., you can
    interpret the ||.|| as |.| in Equation (14).
    
    args:
        x:    torch tensor with shape [batch_size, 13]
        l:    torch tensor with shape [batch_size]
        V:    torch tensor with shape [batch_size]
        dVdt: torch tensor with shape [batch_size]
        dVdx: torch tensor with shape [batch_size, 13] 

    returns:
        h2:   torch tensor with shape [batch_size]
    """
    return torch.abs(torch.min(dVdt + hamiltonian(x, dVdx), l - V))