import numpy as np


def sub_batch(X, sub_batch_size):
    """Applies the sub-batching procedure.

    Transforms a `M x N` 2D dataset `X` into a `B x T x N` 3D
    dataset `S`, where:

    - `M`: number of samples
    - `N`: number of features
    - `T`: sub-batch size (`sub_batch_size`)
    - `B`: number of output sequences, given by `B = N - T + 1`

    Arguments:
        X: 2D dataset to be transformed into a 3D dataset of sequences.
        sub_batch_size: desired sequence length of each sub-batch.

    Returns:
        A 3D batch of sequences.
    """
    num_sub_batches = X.shape[0] - sub_batch_size + 1

    constant_cols = np.expand_dims(np.arange(sub_batch_size), 0)
    constant_rows = np.expand_dims(np.arange(num_sub_batches), 0).T

    indexes = constant_cols + constant_rows
    return X[(indexes)]


def unroll_predictions(y):
    """Reverts a 3D dataset of sequences into a 2D dataset of samples.

    Arguments:
        y: 3D dataset of sequences to be transformed into 2D.

    Returns:
        A 2D batch of predictions.
    """
    y_flipped = np.fliplr(y)
    num_sub_batches, sub_batch_size, _ = y.shape
    offsets = range((sub_batch_size - 1), -num_sub_batches, -1)

    diagonals = [
        np.mean(np.diagonal(y_flipped, offset), axis=-1)
        for offset in offsets
    ]

    return np.array(diagonals)
