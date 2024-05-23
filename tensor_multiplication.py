import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimensions is a tuple of integers.
    """
    # Using torch.ones and multiplying by val
    res = torch.ones(dimensions) * val
    return res

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    # Elementwise product
    res = A * B
    return res

def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W.
    """
    # Matrix product using torch.matmul
    res = torch.matmul(X, W.T)
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W, and add the bias.
    """
    # Matrix product and adding bias
    res = torch.matmul(X, W.T) + b
    return res

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    """
    # Step function using torch.heaviside
    res = torch.heaviside(sum_total, torch.tensor(0.0))
    return res

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    """
    # Compute the matrix product with bias
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    # Apply activation function
    res = calculate_activation(sum_total)
    return res
