# train.py

import torch
import torch.nn as nn
import config

def train(data_loader, model, optimizer, device=config.DEVICE):
    """training the torch model

    Args:
        data_loader ([type]): torch dataloader
        model ([type]): torch model(lstm)
        optimizer ([type]): torch optimizer
        device ([type], optional): can be 'cpu', or 'cuda'. Defaults to config.DEVICE.
    """
    # set model in training mode
    model.train()

    #use data loader to iterate through data
    for data in data_loader:

        # get the X, and y in tensor format from the data loader
        tweets = data['tweets']
        targets = data['targets']

        # move data to device
        tweets = tweets.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # empty the grads
        optimizer.zero_grad()

        # create an output by a forward pass on the model on the input
        out = model(tweets)

        #define loss
        loss = nn.BCEWithLogitsLoss(out, targets)

        # do update the params with the new grads
        optimizer.step()