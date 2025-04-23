import torch.nn as nn
from PatchDiscriminator import PatchDiscriminator
from GanLoss import GANLoss
from Unet_Generator import UNetGenerator
from torch import optim
from init import init_model

import torch

class MainModel(nn.Module):
    """
    MainModel is a class that encapsulates the training process of a GAN-based image colorization model.
    
    Attributes
    ----------
    device : torch.device
        The device (CPU or GPU) on which the model will run.
    lambda_L1 : float
        The weight for the L1 loss in the GAN loss function.
    net_G : nn.Module
        The generator network (Unet) used for generating color images.
    net_D : nn.Module
        The discriminator network (PatchDiscriminator) used for distinguishing real from generated images.
    GANcriterion : GANLoss
        The loss function used for the GAN's adversarial loss.
    L1criterion : nn.L1Loss
        The loss function used for the L1 loss.
    opt_G : torch.optim.Adam
        The optimizer for the generator network.
    opt_D : torch.optim.Adam
        The optimizer for the discriminator network.
    """
    
    def __init__(self, net_G=None, lr_G=1e-4, lr_D=1e-4, 
                 beta1=0, beta2=0.999, lambda_L1=100.):
        """
        Initialize the MainModel with given parameters.

        Parameters
        ----------
        net_G : nn.Module, optional
            An existing generator model to use; if None, a new Unet model will be created.
        lr_G : float
            Learning rate for the generator optimizer.
        lr_D : float
            Learning rate for the discriminator optimizer.
        beta1 : float
            Coefficient for the first moment term in Adam optimizer.
        beta2 : float
            Coefficient for the second moment term in Adam optimizer.
        lambda_L1 : float
            Weight for the L1 loss in the generator's loss function.
        """
        super().__init__()

        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        # Initialize the generator network (Unet) or use the provided one
        if net_G is None:
            self.net_G = init_model(UNetGenerator(input_nc=1,output_nc=2,ngf=64), model_name='Generator', device=self.device)
        else:
            self.net_G = net_G.to(self.device)

        # Initialize the discriminator network (PatchDiscriminator)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), model_name='Discriminator', device=self.device)

        # Initialize loss functions
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()

        # Initialize optimizers for both networks
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        """
        Set the requires_grad attribute for all parameters of a model.

        Parameters
        ----------
        model : nn.Module
            The model whose parameters' requires_grad attribute will be set.
        requires_grad : bool
            Whether to set requires_grad to True or False.
        """
        for p in model.parameters():
            p.requires_grad = requires_grad
    
    def setup_input(self, data):
        """
        Setup input data for the model. This method prepares the input tensors and moves them to the correct device.

        Parameters
        ----------
        data : dict
            A dictionary containing the input data. Expected to have keys 'L' (grayscale image) and 'ab' (real color image).
        """
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        """
        Forward pass through the generator network.

        This method generates fake color images from the input grayscale images.
        """
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        """
        Backward pass for the discriminator network.

        This method computes the loss for the discriminator and performs backpropagation.
        """
        # Create combined images for the discriminator (fake and real)
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())  # Detach to avoid backpropagating through the generator
        self.loss_D_fake = self.GANcriterion(fake_preds, False)  # Loss for fake images

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)  # Loss for real images

        # Average the losses for the discriminator
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()  # Backward pass for discriminator
    
    def backward_G(self):
        """
        Backward pass for the generator network.

        This method computes the loss for the generator and performs backpropagation.
        """
        # Combine the input grayscale with generated color images
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        
        # Calculate losses
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)  # GAN loss for the generator to fool the discriminator
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1  # L1 loss for the generator

        # Total generator loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()  # Backward pass for generator
    
    def optimize(self):
        """
        Perform optimization steps for both the generator and discriminator.

        This method updates the weights of the generator and discriminator networks based on their respective losses.
        """
        # Forward pass for the generator
        self.forward()
        
        # Update discriminator
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()  # Zero gradients for discriminator
        self.backward_D()  # Backward pass for discriminator
        self.opt_D.step()  # Update discriminator parameters
        
        # Update generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)  # Freeze discriminator's parameters
        self.opt_G.zero_grad()  # Zero gradients for generator
        self.backward_G()  # Backward pass for generator
        self.opt_G.step()  # Update generator parameters