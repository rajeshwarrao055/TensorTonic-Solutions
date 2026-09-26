import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    L = max_len if max_len is not None else max((len(row) for row in seqs), default = 0)
    padded = []

    for seq in seqs:
        row = list(seq)[:L]
        row += [pad_value] * (L-len(row))
        padded.append(row)

    arr = np.array(padded)
    if arr.size == 0:
        arr = arr.astype(int)
    return arr.reshape(len(seqs), L)
    