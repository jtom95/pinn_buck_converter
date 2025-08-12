import numpy as np
import torch


# define a dummy discrete differential function

def chose_next_point(x0, a, b):
    dt = 0.1
    x1 = x0 + (a * x0 + b) * dt
    return x1

def inverse_chose_next_point(x1, a, b):
    dt = 0.1
    x0 = (x1 - b*dt) / (a * dt + 1)
    return x0


# true values of a and b
a_true = 2
b_true = 3

# generate a dataset of 5 points
x0 = 0
x1 = chose_next_point(x0, a_true, b_true)
x2 = chose_next_point(x1, a_true, b_true)
x3 = chose_next_point(x2, a_true, b_true)
x4 = chose_next_point(x3, a_true, b_true)
x = np.stack([x0, x1, x2, x3, x4], axis=0)

# verify the backward function
x3_reconstructed = inverse_chose_next_point(x4, a_true, b_true) 
x2_reconstructed = inverse_chose_next_point(x3_reconstructed, a_true, b_true)
x1_reconstructed = inverse_chose_next_point(x2_reconstructed, a_true, b_true)
x0_reconstructed = inverse_chose_next_point(x1_reconstructed, a_true, b_true)

x_reconstructed = np.stack([x0_reconstructed, x1_reconstructed, x2_reconstructed, x3_reconstructed, x4], axis=0)

print("Original x:", x)
print("Reconstructed x:", x_reconstructed)


## GUESS PARAMETERS: 
a_guess = 2.1
b_guess = 2.95

# define the residuals: [(fwd_pred - fwd_obs), (bck_pred - bck_obs)]
def compute_residuals(x, a, b):
    fwd_pred = chose_next_point(x[:-1], a, b)
    bck_pred = inverse_chose_next_point(x[1:], a, b)
    return [(fwd_pred - x[1:]), (bck_pred - x[:-1])]

residuals = compute_residuals(x, a_guess, b_guess)

print("Residuals:")
print(residuals)

print("done")