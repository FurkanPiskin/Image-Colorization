import torch.nn as nn

class DecoderBlock(nn.Module):
    """A decoder block for the U-Net generator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_dropout (bool): Whether to apply dropout.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(DecoderBlock, self).__init__()  # Initialize the parent class
        layers = [
            nn.ReLU(inplace=True),  # ReLU activation for non-linearity
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Transposed convolution for upsampling
            nn.BatchNorm2d(out_channels)  # Batch normalization layer
        ]
        if use_dropout:  # Check if dropout should be applied
            layers.append(nn.Dropout(0.5))  # Add dropout layer with 50% rate
        self.block = nn.Sequential(*layers)  # Create a sequential block of layers

    def forward(self, x):
        """Forward pass through the decoder block."""
        return self.block(x)  # Pass input through the block