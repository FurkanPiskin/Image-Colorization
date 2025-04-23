from torch.utils.data import Dataset,DataLoader
from ColorizationDataset import ColorizationDataset

def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):
    """
    Creates DataLoader instances for training and validation datasets.

    This function sets up the DataLoader to manage batching and loading of the dataset. It helps in efficiently
    fetching batches of images for training or validation.

    Args:
        batch_size (int): The number of images to include in each batch.
        n_workers (int): The number of worker processes to use for data loading.
        pin_memory (bool): Whether to pin memory in GPU to speed up data transfer.
        **kwargs: Additional keyword arguments passed to the ColorizationDataset constructor.

    Returns:
        DataLoader: A DataLoader instance that provides batches of images.
    """
    # Create an instance of ColorizationDataset with the provided arguments
    dataset = ColorizationDataset(**kwargs)
    
    # Create and return a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    
    return dataloader 