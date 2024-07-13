import numpy as np
from trax import layers as tl


def InjectInputs(input_vocab_sizes, d_emb):
    """Injects discrete auxiliary inputs (e.g. time) into the model.

    Args:
        input_vocab_sizes (list[int]): Vocab sizes of the auxiliary input.
            If None, then it is skipped. Example: [30, 7, 24, 512], corresponds
            to a day in month, day in a week, hour, and time series id.
        d_emb: embedding dimension.
    """
    if input_vocab_sizes is None:
        # Ignore the auxiliary inputs.
        return tl.Parallel(
            # context_emb, inputs
            None,
            tl.Drop(),
        )
    else:
        # The model should process the inputs.
        input_embs = [
            tl.Serial(
                tl.Fn("AsInt", lambda x: x.astype(np.int32)),
                tl.Embedding(vocab_size, d_emb),
            )
            if vocab_size is not None
            else tl.Drop()
            for vocab_size in input_vocab_sizes
        ]
        # number of List elements that are not None
        n_items = len([x for x in input_vocab_sizes if x is not None])
        return tl.Serial(
            # context_emb, inputs
            tl.Parallel(
                None,
                tl.Serial(
                    # inputs (batch_size, seq_len, n_inputs)
                    tl.Split(n_items=len(input_vocab_sizes)),
                    # input1 (batch_size, seq_len, 1), input2, ...
                    tl.Parallel(*input_embs),
                    # input1_emb (batch_size, seq_len,1, d_emb),
                    # input2_emb, ...
                    tl.Concatenate(n_items=n_items, axis=2),
                    # input_embs (batch_size, seq_len, n_inputs, d_emb)
                    tl.Sum(axis=2),
                    tl.LayerNorm(),
                    # input_emb (batch_size, seq_len, d_emb)
                ),
            ),
            # context_emb, input_emb
            tl.Add(),
            tl.LayerNorm(),
            # context_and_input_emb
        )
