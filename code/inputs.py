from functools import partial

import gin
import numpy as np
from datasets import Dataset
from index_streams import create_eval_index_stream, create_train_index_stream


def shuffle_decorator(slice_stream, context_length):
    def shuffle_fn(series, inp, target, mask):
        perm = np.random.permutation(context_length)
        def permute_prefix(arr):
            pre = arr[:len(perm)][perm]
            return np.concatenate([pre, arr[len(perm):]])
        return (permute_prefix(series), permute_prefix(inp), permute_prefix(target), mask)
    return (shuffle_fn(*sl) for sl in slice_stream)


def slice_stream(index_stream, dataset, inputs, eval_horizon):
    """ Takes a dataset and iterates over index_stream yielding tuples 
        (series, inp, target, mask).
    Args:
        index_stream: a generator of triples (series_idx, start, end).
        dataset: nd_array of shape (ts_num, ts_len).
        inputs: a function (series_idx, start, end) => ndarray 
        of shape (num_inputs, end - start).
        eval_horizon: how many last dataset timesteps should be masked.
    """
    for sample in index_stream:
        (series_idx, slice_start, slice_stop) = sample
        series = dataset[series_idx][slice_start:slice_stop]
        inp = inputs(series_idx, slice_start, slice_stop)
        mask = np.zeros_like(series)
        mask[(-eval_horizon):] = 1
        yield (series, inp, series, mask)


def minibatch_stream(slice_stream, batch_size, _):
    """ Aggregates samples obtained from slice_stream into batches.
    Args:
        slice_stream: a generator of tuples (series, inp, target, mask).
        batch_size: 0th dimension of each ndarray in the output tuple,
    Returns:
        the output tuple (series, inputs, targets, masks) of batched slices.
    """
    batch = [], [], [], []
    for slice in slice_stream:
        for batch_comp, slice_comp in zip(batch, slice):
            batch_comp.append(slice_comp)

        if len(batch[0]) == batch_size:
            yield [np.stack(batch_comp) for batch_comp in batch]
            batch = [], [], [], []

    # Output the last, partially complete batch.
    if len(batch[0]) > 0:
        padding = batch_size - len(batch[0])
        yield [np.pad(np.stack(batch_comp),
                    ((0, padding),) + ((0, 0), ) * (np.stack(batch_comp).ndim - 1),
                    constant_values=0)
                for batch_comp in batch]


@gin.configurable(module='code.inputs')
def CreateInputs(
    dataset: Dataset,
    batch_size: int = gin.REQUIRED,
    series_length: int = gin.REQUIRED,
    weighted_sampling: bool = gin.REQUIRED,
    full_eval: bool = False,
    traxify: bool = True,  # For testing purposes.
    shuffle: bool = False,
):
    assert dataset.train_data.shape[1] >= series_length, \
        f'Length of input to transformer ({series_length}) should be shorter than \
        train dataset length ({dataset.train_data.shape[1]}).'

    assert dataset.eval_data.shape[1] >= series_length, \
        f'Length of input to transformer ({series_length}) should be shorter than \
        eval dataset length ({dataset.eval_data.shape[1]}).'

    train_data = dataset.train_data
    eval_data = dataset.eval_data
    eval_horizon = dataset.eval_horizon
    covariates = dataset.covariates

    train_index_stream = create_train_index_stream(
        train_data, series_length, weighted_sampling)
    eval_index_stream = create_eval_index_stream(
        eval_data, series_length, full_eval)

    train_slice_stream = slice_stream(
        train_index_stream, train_data, covariates, 0)
    eval_slice_stream = slice_stream(
        eval_index_stream, eval_data, covariates, eval_horizon)

    if shuffle:
        context_length = series_length - eval_horizon
        eval_slice_stream = shuffle_decorator(eval_slice_stream, context_length)

    train_stream = partial(minibatch_stream, train_slice_stream, batch_size)
    eval_stream = partial(minibatch_stream, eval_slice_stream, batch_size)

    if traxify:
        from trax.data import inputs as ti
        return ti.Inputs(train_stream, eval_stream)
    else:
        return (train_stream, eval_stream)
