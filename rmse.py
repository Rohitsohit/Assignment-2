import numpy as np

def rmse(predictions, targets):
    # Convert predictions and targets to numpy arrays
    pred = np.array(predictions)
    tar = np.array(targets)
    
    # Compute the RMSE
    rmse = np.sqrt(np.mean((pred - tar) ** 2))
    
    return rmse
