def derive(f, x, h=0.0001):
    # Calculate the derivative using the central difference formula
    derivative = (f(x + h) - f(x - h)) / (2 * h)
    return derivative
