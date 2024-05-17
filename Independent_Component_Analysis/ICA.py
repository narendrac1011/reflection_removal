import numpy as np
import cv2
import argparse

# Let's find the axis of maximal variance first
# So we need to create a function to find the angle theta
def Theta_Estimation(Y1, Y2, Moment):

    # R(i) = y1(i)^2 + y2(i)^2
    R = Y1**2 + Y2**2

    # Phi(i) = tan_inverse(y2(i)/y1(i))
    PHI = np.arctan2(Y2, Y1)

    # Numerator
    Numerator = np.sum((R**2) * np.sin(Moment * PHI))

    # Denominator
    Denominator = np.sum((R**2) * np.cos(Moment * PHI))

    # Rotation Angle
    Theta = (1 / Moment) * np.arctan2(Numerator, Denominator)
    return Theta


# The scaling of each axis is determined by first computing the variance along the axis of maximal and minimal variance, i.e., the axis oriented at theta_1 and theta_1 - pi/2.
# We need to create a function to find the scaling matrix
def ScalingMatrix_Estimation(Y1, Y2, Theta):

    # Compute the variances along the rotation axes theta and (theta - pi/2)
    # Compute the rotated coordinates
    Scaling_1 = Y1 * np.cos(Theta) + Y2 * np.sin(Theta)
    Scaling_2 = Y1 * np.cos(Theta - (np.pi/2)) + Y2 * np.sin(Theta - (np.pi/2))

    # The Variances 's1' and 's2'
    s1 = np.sum((Scaling_1)**2)
    s2 = np.sum((Scaling_2)**2)

    # Construct the scaling matrix
    Scaling_Matrix = np.diag([1 / s1, 1 / s2])
    return Scaling_Matrix


# We have to now decompose the Matrix M, i.e. the linear mixing matrix
# We know that, M = R1 * S * R2, where R1 and R2 are orthonormal (rotation) matrices and S is a diagonal (scaling) matrix.
# We need to create a function to find this Matrix M
def Decomposition(Y1, Y2):

    # Mean Subtraction from 'Y1' and 'Y2'
    Y1 -= Y1.mean()
    Y2 -= Y2.mean()

    # Theta_1
    Theta_1 = Theta_Estimation(Y1, Y2, 2)
    Theta_2 = Theta_Estimation(Y1, Y2, 4)

    # Rotation Matrix 'R1'
    R1_inv = np.array([[np.cos(Theta_1), -np.sin(Theta_1)], [np.sin(Theta_1), np.cos(Theta_1)]]).transpose()

    # Rotation Matrix 'R2'
    R2_inv = np.array([[np.cos(Theta_2), -np.sin(Theta_2)], [np.sin(Theta_2), np.cos(Theta_2)]]).transpose()

    # Scaling Matrix 'S'
    S_inv = ScalingMatrix_Estimation(Y1, Y2, Theta_1)

    # Linear Mixing Matrix 'M'
    M_inv = np.matmul(R2_inv, np.matmul(S_inv, R1_inv))

    return M_inv