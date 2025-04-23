import torch.nn as nn

class EncoderBlock(nn.Module):
    """An encoder block for the U-Net generator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_batchnorm (bool): Whether to use batch normalization.
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(EncoderBlock, self).__init__()  # Initialize the parent class
        layers = [
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation with negative slope of 0.2
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)  # 2D convolution layer for downsampling
        ]
        if use_batchnorm:  # Check if batch normalization should be applied
            layers.append(nn.BatchNorm2d(out_channels))  # Add batch normalization layer
        self.block = nn.Sequential(*layers)  # Create a sequential block of layers

    def forward(self, x):
        """Forward pass through the encoder block."""
        return self.block(x)  # Pass input through the block