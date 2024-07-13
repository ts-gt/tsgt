"""Decoding with Trax models."""

import numpy as np


def autoregressive_sample(
    model,
    sample_fn,
    context,
    inputs=None,
    batch_size=1,
    start_element=0,
    horizon_length=100,
):
    """Returns a batch of sequences created by autoregressive sampling.

    This function uses `model` to generate outputs one position at a time, with
    access to context for the current position and all preceding positions. The
    new output becomes the next position's input, and this loop repeats until
    either the model outputs the `eos_id` value or the output sequence reaches
    `horizon_length` items.

    Args:
      model: A layer object (subclass of `trax.layers.Layer`) created in
          `'predict'` mode and initialized from trained weights. The model
          must have a structure that allows it to run as autoregressive
          one-sample-at-a-time predictor (e.g., `trax.models.TransformerLM`),
          except if `eval_mode` is set -- any model can be sampled then,
          but the sampling process may be much slower.
      sample_fn: Function that samples the new output from the predicted
          distribution parameters (model's output).
      context: Sequence of symbols the model sees as input the first time it
          generates an output. If None, the model must generate the first output
          with no input to guide it.
      inputs: Sequence of auxiliary inputs (e.g. time) that can help in
          prediction. If None, only the context is fed into the model.
      batch_size: Number of sequences to generate in parallel as a batch.
      start_element: The start symbol (ID/integer) for the autoregressive process,
          or array of shape (`batch_size`, 1) of such integers.
      horizon_length: Length of the generated sequences.

    Returns:
      Tensor of integers with shape (`batch_size`, `horizon_length`) representing
      a batch of output sequences.
    """
    if context is None:
        context = np.empty((batch_size, 0), dtype=np.int32)

    if context.shape[0] != batch_size:
        raise ValueError(
            f"Context batch size ({context.shape[0]}) does not match "
            f"batch_size arg ({batch_size}."
        )

    if np.isscalar(start_element):
        start_symbol = np.full((batch_size, 1), start_element, dtype=np.int32)
    else:
        start_symbol = start_element

    current_symbols = np.concatenate([start_symbol, context], axis=1)

    if inputs is not None:
        if inputs.shape[0] != batch_size:
            raise ValueError(
                f"Inputs batch size ({inputs.shape[0]}) does not match "
                f"batch_size arg ({batch_size}."
            )

        expected_length = context.shape[1] + horizon_length
        if inputs.shape[1] != expected_length:
            raise ValueError(
                f"Invalid length of the inputs: expected {expected_length}, got "
                f"{inputs.shape[1]}."
            )
        input_index = current_symbols.shape[1]
        current_symbols = (current_symbols, inputs[:, :input_index])

    result = []
    for _ in range(horizon_length):
        model_output = model(current_symbols)
        sample = sample_fn(model_output[:, -1, :])[:, None]  # Add the time dimension.
        result.append(sample)

        # The model is autoregressive and in 'predict' mode, so its history is cached
        # in the model state and the next context is the single symbol just sampled.
        if inputs is None:
            current_symbols = sample
        else:
            current_symbols = (sample, inputs[:, input_index : (input_index + 1)])
            input_index += 1

    return np.concatenate(result, axis=1)
