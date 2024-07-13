import itertools
import numpy as np
import pandas as pd
import tqdm


def predict_batches(
    predictor, weights, batches, n_samples
):
    for batch in tqdm.tqdm(list(batches)):
        dataset_batch_size = batch[0].shape[0]  # usually 16
        # Go one-by-one over each element element in the batch.
        for start in range(0, dataset_batch_size):

            subbatch = tuple(item[start:(start+1)] for item in batch)
            # One ground-truth series. mask.shape [1, series].
            (ground_truth_subbatch, _, _, mask) = subbatch
            # If the mask is all 0s, there is nothing to generate
            # (happens e.g., when the batch is partially completed).
            if np.sum(mask) == 0:
                break

            # Repeat each element `n_samples` times (to make that many predictions).
            repeated_subbatch = tuple(
                np.repeat(item, repeats=n_samples, axis=0) for item in subbatch
            )

            pred_subsubbatches = []
            # Split `n_samples` into batches of size `dataset_batch_size`.
            for substart in range(0, n_samples, dataset_batch_size):
                substop = substart + dataset_batch_size
                subsubbatch = tuple(
                    item[substart:substop] for item in repeated_subbatch
                )
                pred_subsubbatch = predict_batch(
                    predictor, weights, subsubbatch
                )
                pred_subsubbatches.append(pred_subsubbatch)
            pred_subbatch = np.concatenate(pred_subsubbatches)

            horizon_length = pred_subbatch.shape[-1]
            ground_truth_subbatch = ground_truth_subbatch[:, -horizon_length:]
            pred_subbatch = np.reshape(
                pred_subbatch, (1, n_samples, horizon_length)
            )

            yield (pred_subbatch, ground_truth_subbatch)


def predict_batch(predictor, weights, batch):
    """Performs evaluation on a single batch."""
    (series, inputs, targets, mask) = batch

    assert np.all((mask == 0) | (mask == 1))
    horizon_lengths = np.sum(mask, axis=1)
    assert horizon_lengths.min() == horizon_lengths.max(), \
        "All slices in a batch should have the same horizon length."
    horizon_length = int(np.mean(horizon_lengths))
    (_, total_length) = mask.shape
    context_length = total_length - horizon_length
    context = series[:, :context_length]

    pred = predictor.predict(
        weights=weights,
        context=context,
        horizon_length=horizon_length,
        inputs=inputs,
    )
    return pred[:, :horizon_length]


def simulate(
    inputs_iterable,  
    weights,
    predictor,
    dataset,
    n_samples,
):
    horizon = dataset.eval_horizon
    predictor.before_training(inputs_iterable)
    inputs_iterable = inputs_iterable()

    eval_stream = inputs_iterable.eval_stream(1)

    # A list containig batches from `eval_task.input`.
    batches = list(eval_stream)

    # Each element from `prediction` generator corresponds
    # to `n_samples` for a single time series. Consequently,
    # the generator provides `n_time_series * n_samples` elements.
    # `predictions_generator` generats tuples of the form
    # (predictions, ground_truth), with shapes
    # (1, n_samples, horizon) and (1, horizon), resp.
    prediction_generator = predict_batches(
        predictor, weights, batches, n_samples
    )

    # Get predictions for all time series in experiment `exp_idx` time window.
    outcome = list(prediction_generator)
    # Number of time series.
    N = len(outcome)
    # Split into predictions and ground truth, with shapes
    # (N, n_samples, horizon) and (N, horizon), resp.
    preds, gts = zip(*outcome)  # (N, 1, n_samples, horizon), (N, 1, horizon)
    preds = np.squeeze(preds)  # (N, n_samples, horizon)
    preds = np.reshape(preds, (-1, horizon))  # (N * n_samples, horizon)
    gts = np.repeat(gts, n_samples, axis=1)  # (N, n_samples, horizon)
    gts = np.concatenate(gts, axis=0)  # (N * n_samples, horizon)
    outcome = np.concatenate([preds, gts], axis=1)  # (N * n_samples, 2 * horizon)
    ts_ids, pred_ids = range(N), range(n_samples)
    # Encode time series ids and prediction ids as a list of tuples
    # of the form (ts_id, pred_id) with shape (N * n_samples, 2).
    ts_pred_ids = np.array(list(itertools.product(ts_ids, pred_ids)))
    # Concatenate time series ids and prediction ids with the predictions
    outcome = np.concatenate(
        [ts_pred_ids, outcome], axis=1
    )  # (N * n_samples, 2 + 2 * horizon)
    column_names = (
        ["ts_id", "pred_id"]
        + [f"pred_{i}" for i in range(horizon)]
        + [f"gt_{i}" for i in range(horizon)]
    )
    simulations_df = pd.DataFrame(outcome, columns=column_names)
    # Add the timestamp.
    simulations_df["eval_start"] = dataset.eval_start
    simulations_df["eval_end"] = dataset.eval_end
    simulations_df["horizon"] = horizon
    return simulations_df.reset_index()
