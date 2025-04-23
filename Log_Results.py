def log_results(loss_meter_dict):
    """
    Print the average values of the recorded losses.

    Parameters
    ----------
    loss_meter_dict : dict
        A dictionary where keys are loss names and values are AverageMeter instances.
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        # Print the average value of each loss, formatted to five decimal places
        print(f"{loss_name}: {loss_meter.avg:.5f}")
        