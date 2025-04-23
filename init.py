from torch import nn
def init_weights(net, init='norm', gain=0.02, name='Generator'):
    """
    Initialize the weights of the network.

    Parameters
    ----------
    net : nn.Module
        The neural network model whose weights need to be initialized.
    init : str
        The initialization method to use. Options are:
        - 'norm': Normal distribution
        - 'xavier': Xavier initialization
        - 'kaiming': Kaiming initialization
    gain : float
        Scaling factor for the initialization. Default is 0.02.
    name : str
        The name of the model (used for print statement). Default is 'Generator'.
    
    Returns
    -------
    nn.Module
        The network with initialized weights.
    """
    
    def init_func(m):
        """
        Initialize weights and biases of the module.

        Parameters
        ----------
        m : nn.Module
            A module (layer) in the network.
        """
        classname = m.__class__.__name__  # Get the class name of the module
        if hasattr(m, 'weight') and 'Conv' in classname:
            # Initialize convolutional layers
            if init == 'norm':
                # Normal distribution initialization
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                # Xavier initialization
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                # Kaiming initialization
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                # Initialize biases to zero
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            # Initialize BatchNorm2d layers
            nn.init.normal_(m.weight.data, 1., gain)  # Mean=1, std=gain
            nn.init.constant_(m.bias.data, 0.)  # Bias to zero
            
    net.apply(init_func)  # Apply the initialization function to all layers in the network
    print(f"{name.capitalize()} model initialized with {init} initialization")

    return net

def init_model(model, model_name, device):
    model = model.to(device)
    model = init_weights(model, name=model_name)
    return model    
