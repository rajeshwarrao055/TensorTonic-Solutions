import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n,d = X.shape
    w = np.zeros(d)
    b = 0.0
    loss = 0

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        err = p - y

        dw = X.T @ err / n
        db = err.mean()

        w -= lr * dw
        b -= lr * db

    return w,b