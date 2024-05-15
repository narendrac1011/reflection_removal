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