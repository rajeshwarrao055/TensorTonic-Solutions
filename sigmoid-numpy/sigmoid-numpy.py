import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    if(isinstance(x, float) or isinstance(x, int)):
        return 1/(1 + np.exp(-x))
    else:
        return [sigmoid(val) for val in x]
    pass