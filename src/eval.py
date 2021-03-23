# eval.py

import torch
import torch.nn as nn
import config

def eval(data_loader, model, device=config.DEVICE):
    """evaluating the torch model

    Args:
        data_loader ([type]): torch dataloader
        model ([type]): torch model(lstm)
        device ([type], optional): can be 'cpu', or 'cuda'. Defaults to config.DEVICE.
    """
    final_predictions = []
    final_targets = []

     # set model in evaluation mode
    model.eval()

    # use context manager to stop updating and do gradient calculation
    with torch.no_grad():
        #use data loader to iterate through data
        for data in data_loader:

            # get the X, and y in tensor format from the data loader
            tweets = data['tweets']
            targets = data['targets']

             # move data to device
            tweets = tweets.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # create a prediction by a forward pass on the model on the input
            out = model(tweets)

            # move the preduictions and true targets to list and to cpu
            out = out.cpu().numpy().tolist()
            targets = data['targets'].cpu().numpy().tolist()
            final_predictions.extend(out)
            final_targets.extend(targets)
    
    
    return final_predictions, final_targets




