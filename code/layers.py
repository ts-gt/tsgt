from jax import numpy as jnp
import numpy as np
from trax import layers as tl
from trax.layers import initializers as init
from trax import fastmath, shapes
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import initializers as init
from trax.layers.assert_shape import assert_shape


class CausalConv(tl.Conv):
    """Causal (masked) convolution for [batch x time x depth] sequences.
    Maintains causality along time axis. Used in language modeling tasks.
    """

    def __init__(
        self,
        filters,
        kernel_width=3,
        kernel_initializer=None,
        bias_initializer=init.RandomNormalInitializer(1e-6),
        use_bias=True,
        mode='train',
    ):
        super().__init__(
            filters=filters,
            kernel_size=(kernel_width,),
            strides=None,
            padding='VALID',
            dimension_numbers=('NWC', 'WIO', 'NWC'),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=use_bias)
        self._mode = mode

    def init_weights_and_state(self, input_signature):
        """Initializes this layer for fast inference, if in ``'predict'`` mode."""
        super().init_weights_and_state(input_signature)
        if self._mode == 'predict':
            (batch_size, _, depth) = input_signature.shape
            self.state = jnp.zeros((batch_size, self._kernel_size[0], depth))

    def forward(self, x):
        assert self._padding == 'VALID'
        if self._mode in ('train', 'eval'):
            # Left pad with 0s. Applying an unmasked valid convolution on top of this
            # yields a causal convolution.
            pad = self._kernel_size[0] - 1
            x = jnp.pad(x, pad_width=[[0, 0], [pad, 0], [0, 0]], mode='constant')
        else:
            assert self._mode == 'predict'
            n_tokens = x.shape[1]
            # Append the current input to the buffer.
            x = jnp.concatenate((self.state, x), axis=1)
            # Last kernel_size tokens go back to the buffer.
            self.state = x[:, -self._kernel_size[0]:]
            # The sequence fed to the convolution layer is the last (kernel_size - 1)
            # tokens from the buffer plus the current input.
            x = x[:, -(self._kernel_size[0] + n_tokens - 1):]

        return super().forward(x)


@assert_shape('...d->...d')
class DigitEncoding(base.Layer):
    """Implements bare digit encoding.

    Positional encoding includes a kind of dropout, if the layer is created in
    ``'train'`` mode with a nonzero ``dropout`` value. For such a layer, on each
    forward pass a subset of sequence positions selected at random will *not*
    receive positional marking.
    """

    def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
                use_bfloat16=False, precision=None, mode='train',
                kernel_initializer=init.GlorotUniformInitializer()):
        """Creates a :py:class:`PositionalEncoding` instance in a given mode.

        Args:
        max_len: Maximum input sequence length.
        dropout: Probability of *not* adding positional encoding to a sequence
            position. Applies only if layer is created in ``'train'`` mode.
        dropout_broadcast_dims: Axes along which dropout mask values are
            broadcast rather than individually set at random.
        use_bfloat16: If ``True``, use bfloat16 weights instead of the default
            float32; this can save memory but may (rarely) lead to numerical issues.
        precision: int or None; precision in representation used in
            discretization scheme.
        mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
        """
        super().__init__()
        self._max_len = max_len
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')
        if precision is None:
            raise ValueError('Precision must be specified.')
        if mode == 'train':
            self._dropout = dropout
        else:
            self._dropout = 0.0
        self._dropout_broadcast_dims = dropout_broadcast_dims
        self._use_bfloat16 = use_bfloat16
        self._mode = mode
        self._kernel_initializer = kernel_initializer
        self._precision = precision

    def forward(self, x):
        """Returns the input activations, with added positional information.
        Args:
        x: Tensor of token ids with shapes [batch, n_ctx, d_feature].

        Returns:
        Tensor of embedding vectors
        """
        weights = self.weights
        if len(weights.shape) < 3:  # old checkpoints have 1 in first dim already
            weights = weights[None, :, :]  # [1, self._max_len, d_feature]


        if self._mode != 'predict':
            token_ids = jnp.arange(x.shape[1])
        else:
            # State in this class is only used for fast inference. In that case,
            # the model is called with consecutive elements position-by-position.
            # This positional encoding layer stores the index of the current
            # position and increments it on each call.
            # Need dymnamic slice to avoid jit error, see
            # https://github.com/google/jax/discussions/7831
            token_ids = jnp.arange(self._max_len)
            token_ids = fastmath.dynamic_slice_in_dim(
                token_ids, self.state, x.shape[1], axis=0)

        digit_ids = jnp.mod(token_ids, self._precision)
        emb = jnp.take(weights, digit_ids, axis=1)  # shape [batch, ids, d_feature]

        # Add dropout if in training mode: drop out some positions from embedding.
        if self._mode != 'predict':
            symbol_size = jnp.shape(x)[1]
            if self._dropout > 0:
                noise_shape = list(emb.shape)
                for dim in self._dropout_broadcast_dims:
                    noise_shape[dim] = 1
                keep_prob = 1.0 - self._dropout
                keep = fastmath.random.bernoulli(self.rng, keep_prob,
                                                tuple(noise_shape))
                multiplier = keep.astype(x.dtype) / keep_prob
                emb = emb * multiplier
        else:  # self._mode == 'predict':
            if self._dropout != 0:
                raise ValueError(f'In predict mode, but dropout rate '
                                f'({self._dropout}) is not zero.')
            # Update cached position.
            self.state += x.shape[1]

        return x + emb


    def init_weights_and_state(self, input_signature):
        """Randomly initializes the digit encoding vectors.
        input_signature shape [batch, n_ctx, d_feature]
        """
        d_feature = input_signature.shape[-1]
        shape_w = (self._precision, d_feature)
        w = self._kernel_initializer(shape_w, self.rng)
        self.weights = w
        if self._mode == 'predict':
            self.state = jnp.zeros((), dtype=jnp.int32)


@assert_shape('...d->...d')
class PositionalDigitEncoding(base.Layer):
    """Implements bare digital positional encoding.

    Positional encoding includes a kind of dropout, if the layer is created in
    ``'train'`` mode with a nonzero ``dropout`` value. For such a layer, on each
    forward pass a subset of sequence positions selected at random will *not*
    receive positional marking.
    """

    def __init__(self, max_len=2048, dropout=0.0, dropout_broadcast_dims=(-2,),
            use_bfloat16=False, start_from_zero_prob=1.0,
            max_offset_to_add=0, d_feature=None, d_digit=None,
            precision=None, mode='train'):
        """Creates a :py:class:`PositionalEncoding` instance in a given mode.

        Args:
        max_len: Maximum input sequence length.
        dropout: Probability of *not* adding positional encoding to a sequence
            position. Applies only if layer is created in ``'train'`` mode.
        dropout_broadcast_dims: Axes along which dropout mask values are
            broadcast rather than individually set at random.
        use_bfloat16: If ``True``, use bfloat16 weights instead of the default
            float32; this can save memory but may (rarely) lead to numerical issues.
        start_from_zero_prob: how often to start from 0 during training,
            (if 1.0, we always start from position 0, if less, we randomize).
        max_offset_to_add: maximum offset to add to the positions during training
            when randomizing; this offset plus input length must still be less than
            max_len for all training examples.
        d_feature: int or None; have this dimension for embeddings + shared FF if
            not None.
        d_digit: int or None; dimension corresponding to digit positional
            encoding, usually a small number since precision is rather small int.
        precision: int or None; precision in representation used in
            discretization scheme.
        mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
        """
        super().__init__()
        self._max_len = max_len
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')
        if [d_digit, precision].count(None) == 1:
            raise ValueError('Both d_digit and precision must be specified or None.')
        if mode == 'train':
            self._dropout = dropout
        else:
            self._dropout = 0.0
        self._dropout_broadcast_dims = dropout_broadcast_dims
        self._use_bfloat16 = use_bfloat16
        self._start_from_zero_prob = start_from_zero_prob
        self._max_offset_to_add = max_offset_to_add
        self._mode = mode
        self._d_feature = d_feature
        self._d_digit = d_digit if d_digit else 0
        self._precision = precision

    def forward(self, inputs):
        """Returns the input activations, with added positional information."""
        weights = self.weights
        if self._d_feature is not None and self._mode != 'predict':
            weights, ff = weights
            weights = jnp.dot(weights[:inputs.shape[1], :], ff)
        if len(weights.shape) < 3:  # old checkpoints have 1 in first dim already
            weights = weights[None, :, :]  # [1, self._max_len, d_feature]
        if self._mode != 'predict':
            x = inputs
            symbol_size = jnp.shape(x)[1]
            if self._mode != 'train' or self._start_from_zero_prob >= 1.0:
                px = weights[:, :symbol_size, :]
            else:
                rng1, rng2 = fastmath.random.split(self.rng, 2)
                start = fastmath.random.randint(rng1, (), 0, self._max_offset_to_add)
                start_from_zero = fastmath.random.uniform(rng2, (), jnp.float32, 0, 1)
                start = jnp.where(start_from_zero < self._start_from_zero_prob,
                                  jnp.zeros((), dtype=jnp.int32), start)
                px = fastmath.dynamic_slice_in_dim(weights, start, symbol_size, axis=1)
            # TODO: dropout on digits as well?
            if self._dropout == 0:
                return x + px
            else:
                noise_shape = list(px.shape)
                for dim in self._dropout_broadcast_dims:
                    noise_shape[dim] = 1
                keep_prob = 1.0 - self._dropout
                keep = fastmath.random.bernoulli(self.rng, keep_prob, tuple(noise_shape))
                multiplier = keep.astype(x.dtype) / keep_prob
                return x + px * multiplier
        else:
            if self._dropout != 0:
                raise ValueError(f'In predict mode, but dropout rate '
                                 f'({self._dropout}) is not zero.')

            # State in this class is only used for fast inference. In that case,
            # the model is called with consecutive elements position-by-position.
            # This positional encoding layer stores the index of the current
            # position and increments it on each call.
            emb = fastmath.dynamic_slice_in_dim(
                    weights, self.state, inputs.shape[1], axis=1)
            self.state += inputs.shape[1]
            return inputs + emb

    def init_weights_and_state(self, input_signature):
        """Randomly initializes the positional encoding vectors.

        Args:
            input_signature: :py:class:`ShapeDtype` instance characterizing the input
                this layer should compute on.
        """
        d_feature = input_signature.shape[-1]
        if self._d_feature is not None:
            d_feature = self._d_feature
        # If precision is None, self._d_digit is 0 and we don't add digit encoding.
        d_position = d_feature - self._d_digit
        pe = np.zeros((self._max_len, d_position), dtype=np.float32)
        position = np.arange(0, self._max_len)[:, np.newaxis]
        div_term = np.exp(
                np.arange(0, d_position, 2) * -(np.log(10000.0) / d_position))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        if self._precision is not None:
            d_digit, precision = self._d_digit, self._precision
            div_term_digit = 2**(np.arange(0, d_digit//2)) * 2 * np.pi / precision
            pe_digit = np.zeros((self._max_len, d_digit), dtype=np.float32)
            pe_digit[:, 0::2] = np.sin(position * div_term_digit)
            pe_digit[:, 1::2] = np.cos(position * div_term_digit)
            pe = np.concatenate([pe_digit, pe], axis=-1)

        if self._use_bfloat16:
            pe = pe.astype(jnp.bfloat16)
        w = jnp.array(pe)  # Trainable parameters, initialized above.
        if self._d_feature is not None:
            ff = init.GlorotUniformInitializer()(
                    (d_feature, input_signature.shape[-1]), self.rng)
            self.weights = w, ff
        else:
            self.weights = w
        if self._mode == 'predict':
            self.state = jnp.zeros((), dtype=jnp.int32)


def Unsqueeze():
    """Returns a layer that adds a dimension to a tensor."""
    layer_name = 'Unsqueeze'

    def f(x):
        return jnp.expand_dims(x, -1)

    return tl.Fn(layer_name, f)


class Stack(tl.base.Layer):
    """Stacks a number of tensors into a single tensor."""

    def __init__(self, n_items=2, axis=-1):
        name = 'Stack' if axis == -1 else f'Stack_axis{axis}'
        super().__init__(n_in=n_items, name=name)
        self._n_items = n_items
        self._axis = axis

    def forward(self, xs):
        return jnp.stack(xs, self._axis)
