"""
AA 276 Homework 1 | Coding Portion | Part 1 of 3


OVERVIEW

In this file, you will implement several functions required by the 
neural CBF library developed by the REALM Lab at MIT to
automatically learn your own CBFs for a 13D quadrotor system!

From this exercise, you will hopefully better understand the course
materials through a concrete example, appreciate the advantages
(and disadvantages) of learning a CBF versus manually constructing
one, and get some hands-on coding experience with using state-of-the-art
tools for synthesizing safety certificates, which you might find
useful for your own work.

If you are interested in learning more, you can find the library
here: https://github.com/MIT-REALM/neural_clbf


INSTRUCTIONS

Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check1.py`.
After the tests pass, train a neural CBF (in your VM) by running `python scripts/train.py`.


IMPORTANT NOTES ON TRAINING
The training can take a substantial amount of time to complete [~9 hours ~= $10].
However, you should be able to implement all code for Parts 1, 2, and 3 in the meantime.
After each training epoch [50 total], the CBF model will save to 'outputs/cbf.ckpt'.
As long as you have at least one checkpoint saved [~10 minutes], Part 3 will load this checkpoint.
Try your best to not exceed $10 in credits -  you can stop training early if you reach this budget limit.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""

import numpy as np
import torch
import pdb

def state_limits():
        """
        Return a tuple (upper, lower) describing the state bounds for the system.
        
        returns:
            (upper, lower)
                where upper: torch float32 tensor with shape [13]
                      lower: torch float32 tensor with shape [13]
        """
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
        
        return (upper,lower)
        pass


def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [4]
                  lower: torch float32 tensor with shape [4]
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
    return (upper, lower)
    pass


"""Note: the following functions operate on batched inputs.""" 


def safe_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the prescribed safe set.

    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_safe: torch bool tensor with shape [batch_size]
    """
    is_safe = torch.sqrt(x[:,0]**2 + x[:,1]**2) > 2.8
    return is_safe

    pass


def failure_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the failure set.

    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_failure: torch bool tensor with shape [batch_size]
    """
    is_failure = torch.sqrt(x[:,0]**2 + x[:,1]**2) < 0.5
    return is_failure
    pass

def f(x):
    """
    Return the control-independent part of the control-affine dynamics.
    Note: we have already implemented this for you!

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

def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    # control dims
    Fi = 0
    axi = 1
    ayi = 2
    azi = 3

    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    g = torch.zeros(x.shape[0], x.shape[1], 4)
    g[:,VXi,Fi] = 2*(QW*QY + QX*QZ)
    g[:,VYi,Fi] = 2*(QY*QZ - QW*QX)
    g[:,VZi,Fi] = 2*(1/2 - QX**2 - QY**2)
    g[:,WXi,axi] = 1
    g[:,WYi,ayi] = 1
    g[:,WZi,azi] = 1
    return g
    pass
