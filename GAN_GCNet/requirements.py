import torch.nn as nn

class Interpolate(nn.Module):
    """
    A module for performing interpolation on input tensors.
    
    Args:
        scale_factor (float or Tuple[float]): The scale factor for interpolation.
        mode (str): The interpolation mode. Supported modes: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'.
        align_corners (bool, optional): Whether to align corners of the input and output tensors. Default is True.
    """
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    """
    Perform interpolation on the input tensor x. 
    Args: x (torch.Tensor): The input tensor to be interpolated.
    Returns: torch.Tensor: The interpolated tensor.
    """
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

        return x