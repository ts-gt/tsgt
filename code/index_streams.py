import numpy as np
from random import randrange


def create_full_eval_index_stream(dataset, series_length):
    """Pass through the dataset once, in order."""
    for series_idx in range(len(dataset)):
        slice_start = -series_length
        slice_stop = None
        yield (series_idx, slice_start, slice_stop)


def create_random_eval_index_stream(dataset, series_length):
    """Sample batches indefinitely."""
    while True:
        series_idx = randrange(0, len(dataset))
        slice_start = -series_length
        slice_stop = None
        yield (series_idx, slice_start, slice_stop)


def create_eval_index_stream(dataset, series_length, full_eval):
    if full_eval:
        stream = create_full_eval_index_stream(dataset, series_length)
    else:
        stream = create_random_eval_index_stream(dataset, series_length)

    yield from stream


def get_weights(dataset: np.ndarray, series_length: int) -> np.ndarray:
    ts_num = dataset.shape[0]
    first_nonzeros = (dataset != 0).argmax(axis=1)
    weights = dataset.copy()
    for i in range(1, series_length):
        weights[:, :(-i)] += dataset[:, i:]
    weights = weights / series_length + 1
    for i, j in zip(range(ts_num), first_nonzeros):
        weights[i, :j] = 0
    return weights[:, : (-series_length + 1)]


def create_weighted_train_index_stream(dataset, series_length):
    weights = get_weights(dataset, series_length)
    cum_weights = np.cumsum(weights.flatten())
    ts_len = weights.shape[1]
    # Sample indices indefinitely
    while True:
        x = np.random.uniform(0, cum_weights[-1])
        idx = np.searchsorted(cum_weights, x)
        series_id = idx // ts_len
        slice_start = idx % ts_len
        slice_stop = slice_start + series_length
        yield series_id, slice_start, slice_stop


def create_uniform_train_index_stream(dataset, series_length):
    """Sample batches indefinitely."""
    while True:
        series_idx = randrange(0, len(dataset))
        slice_start = randrange(0, dataset.shape[1] - series_length)
        slice_stop = slice_start + series_length
        yield (series_idx, slice_start, slice_stop)


def create_train_index_stream(dataset, series_length, weighted_sampling):
    """
    Args:
        dataset: ndarray of shape (ts_num, ts_len),
        series_length: positive int, difference between start and end index.
        weighted_sampling: a bool value, if True sample according to weights propotional to the sum of values,
    """
    if weighted_sampling:
        create_stream_fn = create_weighted_train_index_stream
    else:
        create_stream_fn = create_uniform_train_index_stream

    yield from create_stream_fn(dataset, series_length)
