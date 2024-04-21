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
    Scaling_1 = (Y1**2) * (np.cos(Theta))**2 + 2 * Y1 * Y2 * (np.cos(Theta)) * (np.sin(Theta)) + (Y2**2) * (np.sin(Theta))**2

    Scaling_2 = (Y1**2) * (np.cos(Theta - (np.pi/ .2)))**2 + 2 * Y1 * Y2 * (np.cos(Theta - (np.pi/ .2))) * (np.sin(Theta - (np.pi/ .2))) + (Y2**2) * (np.sin(Theta - (np.pi/ .2)))**2

    # The Variances 's1' and 's2'
    s1 = np.sum(Scaling_1)
    s2 = np.sum(Scaling_2)

    # Construct the scaling matrix
    Scaling_Matrix = np.diag([1. / s1, 1. / s2])
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


# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description = "Reflection Removal using ICA")
    parser.add_argument("-i1", "--image1", required = True, help = "Input image 1")
    parser.add_argument("-i2", "--image2", required = True, help = "Input image 2")
    return parser.parse_args()

# Main
if __name__ == "__main__":
    args = parse_arguments()
    i1 = args.image1
    i2 = args.image2
    image1 = cv2.imread(i1).astype(np.float32)
    image2 = cv2.imread(i2).astype(np.float32)

# Resize the images to make sure that their shape is standard
image1 = cv2.resize(image1, (1024, 1024))
image2 = cv2.resize(image2, (1024, 1024))

# Get the Linear Mixing Matrix from the images
M = Decomposition(image1, image2)

# Apply the decomposition to the input images
Im = np.concatenate([image1.reshape(1, -1), image2.reshape(1, -1)], axis=0)
Im = np.matmul(M, Im)

# Reshape and normalize the separated images
i1 = Im[0, :]
i2 = Im[1, :]

i1, i2 = i1 - i1.min(), i2 - i2.min()
i1, i2 = i1 * 255. / i1.max(), i2 * 255. / i2.max()

# Convert the separated images back to uint8
h1 = i1.reshape(image1.shape).clip(0, 255).astype(np.uint8)
h2 = i2.reshape(image2.shape).clip(0, 255).astype(np.uint8)

cv2.imwrite('1-ICA.png', h1)
cv2.imwrite('2-ICA.png', h2)