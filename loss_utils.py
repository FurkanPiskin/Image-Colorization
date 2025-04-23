class AverageMeter:
    """
    A utility class to keep track of average values.

    Attributes
    ----------
    count : float
        The total number of values added.
    avg : float
        The average of the values.
    sum : float
        The sum of the values added.
    """
    
    def __init__(self):
        """
        Initialize the AverageMeter by resetting it.
        """
        self.reset()  # Calls the reset method to initialize the attributes
        
    def reset(self):
        """
        Reset all the statistics to zero.
        """
        self.count, self.avg, self.sum = [0.] * 3
        # Initializes count, avg, and sum to 0.0
    
    def update(self, val, count=1):
        """
        Update the statistics with a new value.

        Parameters
        ----------
        val : float
            The new value to add.
        count : int
            The number of times this value is added (default is 1).
        """
        self.count += count  # Adds the count of new values
        self.sum += count * val  # Adds the total sum of new values
        self.avg = self.sum / self.count  # Updates the average
        
def create_loss_meters():
    """
    Create and return a dictionary of AverageMeter instances for tracking various losses.

    Returns
    -------
    dict
        A dictionary where keys are loss names and values are AverageMeter instances.
    """
    # Creates an AverageMeter for each type of loss to track during training
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,  # Dictionary keys are loss names
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}     

def update_losses(model, loss_meter_dict, count):
    """
    Update the loss meters with the current losses from the model.

    Parameters
    ----------
    model : nn.Module
        The model instance from which to fetch loss values.
    loss_meter_dict : dict
        A dictionary where keys are loss names and values are AverageMeter instances.
    count : int
        The count to use when updating the loss meters.
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)  # Get the loss value from the model using attribute name
        loss_meter.update(loss.item(), count=count)  # Update the AverageMeter with the loss value        
