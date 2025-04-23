from Encoder import EncoderBlock
from Decoder import DecoderBlock
import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """U-Net generator architecture.

    Args:
        input_nc (int): Number of input channels.
        output_nc (int): Number of output channels.
        ngf (int): Number of filters in the first layer.
    """ 
    def __init__(self, input_nc, output_nc, ngf,EncoderBlock_Class=EncoderBlock,DecoderBlock_Class=DecoderBlock):
        super(UNetGenerator, self).__init__()  # Initialize the parent class
        
        # Encoder layers
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)  # Initial convolution layer
        self.e2 = EncoderBlock_Class(ngf, ngf * 2)  # First encoder block
        self.e3 = EncoderBlock_Class(ngf * 2, ngf * 4)  # Second encoder block
        self.e4 = EncoderBlock_Class(ngf * 4, ngf * 8)  # Third encoder block
        self.e5 = EncoderBlock_Class(ngf * 8, ngf * 8)  # Fourth encoder block
        self.e6 = EncoderBlock_Class(ngf * 8, ngf * 8)  # Fifth encoder block
        self.e7 = EncoderBlock_Class(ngf * 8, ngf * 8)  # Sixth encoder block
        self.e8 = EncoderBlock_Class(ngf * 8, ngf * 8, use_batchnorm=False)  # Last encoder block without batch normalization
        
        # Decoder layers
        self.d1 = DecoderBlock_Class(ngf * 8, ngf * 8, use_dropout=True)  # First decoder block with dropout
        self.d2 = DecoderBlock_Class(ngf * 8 * 2, ngf * 8, use_dropout=True)  # Second decoder block with concatenated skip connection
        self.d3 = DecoderBlock_Class(ngf * 8 * 2, ngf * 8, use_dropout=True)  # Third decoder block with concatenated skip connection
        self.d4 = DecoderBlock_Class(ngf * 8 * 2, ngf * 8)  # Fourth decoder block with concatenated skip connection
        self.d5 = DecoderBlock_Class(ngf * 8 * 2, ngf * 4)  # Fifth decoder block with concatenated skip connection
        self.d6 = DecoderBlock_Class(ngf * 4 * 2, ngf * 2)  # Sixth decoder block with concatenated skip connection
        self.d7 = DecoderBlock_Class(ngf * 2 * 2, ngf)  # Seventh decoder block with concatenated skip connection
        self.d8 = nn.Sequential(  # Final output layer
            nn.ReLU(inplace=True),  # ReLU activation for output
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),  # Transposed convolution to produce output
            nn.Tanh()  # Tanh activation function for normalized output
        )

    def forward(self, x):
        """Forward pass through the entire U-Net generator."""
        # Encoder pass
        e1 = self.e1(x)  # Apply the first encoder layer
        e2 = self.e2(e1)  # Apply the second encoder layer
        e3 = self.e3(e2)  # Apply the third encoder layer
        e4 = self.e4(e3)  # Apply the fourth encoder layer
        e5 = self.e5(e4)  # Apply the fifth encoder layer
        e6 = self.e6(e5)  # Apply the sixth encoder layer
        e7 = self.e7(e6)  # Apply the seventh encoder layer 
        e8 = self.e8(e7)  # Apply the eighth encoder layer
        
        # Decoder pass with skip connections
        d1 = self.d1(e8)  # Apply the first decoder layer
        d1 = torch.cat([d1, e7], 1)  # Concatenate with the skip connection from encoder
        d2 = self.d2(d1)  # Apply the second decoder layer
        d2 = torch.cat([d2, e6], 1)  # Concatenate with the skip connection from encoder
        d3 = self.d3(d2)  # Apply the third decoder layer
        d3 = torch.cat([d3, e5], 1)  # Concatenate with the skip connection from encoder
        d4 = self.d4(d3)  # Apply the fourth decoder layer
        d4 = torch.cat([d4, e4], 1)  # Concatenate with the skip connection from encoder
        d5 = self.d5(d4)  # Apply the fifth decoder layer
        d5 = torch.cat([d5, e3], 1)  # Concatenate with the skip connection from encoder
        d6 = self.d6(d5)  # Apply the sixth decoder layer
        d6 = torch.cat([d6, e2], 1)  # Concatenate with the skip connection from encoder
        d7 = self.d7(d6)  # Apply the seventh decoder layer
        d7 = torch.cat([d7, e1], 1)  # Concatenate with the skip connection from encoder
        d8 = self.d8(d7)  # Apply the final output layer
        
        return d8  # Return the output of the generator    