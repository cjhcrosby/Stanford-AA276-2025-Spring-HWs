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

GRAV = 10
LENG = 1
MASS = 2

def state_limits():
        """
        Return a tuple (upper, lower) describing the state bounds for the system.
        
        returns:
            (upper, lower)
                where upper: torch float32 tensor with shape [13]
                      lower: torch float32 tensor with shape [13]
        """
        theta_max = torch.pi
        theta_dot_max = 10.0
        upper = torch.tensor([theta_max, theta_dot_max])
        lower = torch.tensor([-theta_max, -theta_dot_max])
        
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
    u_max = 3
    upper = torch.tensor([u_max])
    lower = torch.tensor([-u_max])
    return (upper, lower)
    pass


"""Note: the following functions operate on batched inputs.""" 


def safe_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the prescribed safe set.
    safe: \mathcal{C} = \set{\mathbf{x} \mid 1 - 100 \theta^{2} - 25 \dot{\theta}^{2} \ge 0}
    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_safe: torch bool tensor with shape [batch_size]
    """
    is_safe = 1 - 100 * x[:,0]**2 - 25 * x[:,1]**2 >= 0
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
    is_failure = torch.abs(x[:,0]) >= 0.3
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
    f[:, 0] = x[:,1]
    f[:, 1] = GRAV/LENG * torch.sin(x[:,0])
    return f

def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """


    g[:,0] = 0
    g[:,1] = 1/(MASS*LENG**2)

    return g
