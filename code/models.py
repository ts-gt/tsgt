import gin
from trax import layers as tl
from trax.layers.assert_shape import assert_shape

import layers
import attention


@gin.configurable(module='code.models')
def TransformerBody(
    d_model=256,
    d_ff_mul=2,
    n_layers=2,
    n_heads=2,
    max_len=2048,
    dropout=0.1,
    conv_activation=None,
    conv_kernel=1,
    fraction_to_rotate=0.25,
    conv_attention_kernel_width=None,
    expand_heads=False,
    mode='train',
    precision=None,
    digit_encoding=False,
    ff_activation=tl.Relu,
):
  """Returns the body of a Transformer decoder model.

  A version of TransformerLM without the input embedding and output softmax
  layers.

  Supports initial convolution.

  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model.
    d_ff_mul: Last/innermost dimension multiplier of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_layers: Number of decoder blocks. Each block includes attention, dropout,
        residual, layer-norm, feedforward (:py:class:`Dense`), and activation
        layers.
    n_heads: Number of attention heads.
    max_len: Maximum length of the sequence for positional encoding.
    dropout: Probability of dropping an activation value when applying dropout
        within decoder blocks. The same rate is also used for attention dropout
        in decoder blocks.
    conv_activation: Activation function for the initial convolution.
    conv_kernel: Kernel size for the initial convolution.
    expand_heads: Whether to expand the feature dimension times n_heads before
        the attention layer.
    mode: If ``'predict'``, use fast inference. If ``'train'``, each decoder
        block will include dropout; else, it will pass all values through
        unaltered.
    precision: int or None; precision in representation used in
        discretization scheme.
    ff_activation: Callable which returns a layer that computes an activation
        function in FFB part of transformer block.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, mode=mode)

  def _DecBlock():
    return _DecoderBlock(
        d_model, d_ff_mul * d_model, n_heads, dropout, dropout_shared_axes=None,
        mode=mode, ff_activation=ff_activation,
        conv_attention_kernel_width=conv_attention_kernel_width,
        fraction_to_rotate=fraction_to_rotate,
        expand_heads=expand_heads,
    )

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      _Dropout(),

      layers.CausalConv(d_model, kernel_width=conv_kernel, mode=mode),
      *([conv_activation()] if conv_activation is not None else []),

      *([layers.DigitEncoding(max_len=max_len,
                           mode=mode,
                           precision=precision)] if digit_encoding else []),
      [_DecBlock() for _ in range(n_layers)],
      tl.LayerNorm(),
  )


@gin.configurable(module='code.models')
def ConvTransformerLM(
    vocab_size,
    conv_activation=None,
    conv_kernel=1,
    d_model=256,
    d_ff_mul=2,
    n_layers=2,
    n_heads=2,
    max_len=2048,
    dropout=0.1,
    dropout_shared_axes=None,
    conv_attention_kernel_width=None,
    mode='train',
    ff_activation=tl.Relu,
    expand_heads=False,
    precision=None,
    digit_encoding=True,
):
  """Returns a convolutional Transformer language model.

  This model performs autoregressive language modeling:

    - input: Array representing a batch of text strings via token IDs
      plus padding markers; shape is (batch_size, sequence_length). Array
      elements are integers in ``range(vocab_size)``, and 0 values mark padding
      positions.

    - output: 3-D array of raw activations with last/innermost dimension of
      ``vocab_size``, suitable for decoding into a batch of token strings;
      shape is (batch_size, sequence_length, ``vocab_size``).

  This model uses only the decoder part of the overall Transformer.

  Args:
    vocab_size: Input vocabulary size -- each element of the input array
        should be an integer in ``range(vocab_size)``. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff_mul: Last/innermost dimension multiplier of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_layers: Number of decoder blocks. Each block includes attention, dropout,
        residual, layer-norm, feedforward (:py:class:`Dense`), and activation
        layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within decoder blocks. The same rate is also
        used for attention dropout in decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'predict'``, use fast inference. If ``'train'``, each decoder
        block will include dropout; else, it will pass all values through
        unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of :py:class:`Layer`.
    precision: int or None; precision in representation used in
        discretization scheme.

  Returns:
    A Transformer language model that maps strings (represented as token ID
    sequences) to sequences of raw (non-normalized) activation vectors; each
    vector in the sequence can be mapped (e.g., by `argmax`) to a token ID.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _DecBlock():
    return _DecoderBlock(
        d_model, d_ff_mul * d_model, n_heads, dropout, dropout_shared_axes=dropout_shared_axes,
        mode=mode, ff_activation=ff_activation,
        conv_attention_kernel_width=conv_attention_kernel_width,
        expand_heads=expand_heads,
    )

  return tl.Serial(
      tl.ShiftRight(mode=mode),
      tl.Embedding(vocab_size, d_model),
       _Dropout(),

      layers.CausalConv(d_model, kernel_width=conv_kernel, mode=mode),
      *([conv_activation()] if conv_activation is not None else []),

      *([layers.DigitEncoding(max_len=max_len,
                           mode=mode,
                           precision=precision)] if digit_encoding is not None else []),
      [_DecBlock() for _ in range(n_layers)],
      tl.LayerNorm(),
      tl.Dense(vocab_size),
  )


def _DecoderBlock(
    d_model,
    d_ff,
    n_heads,
    dropout,
    dropout_shared_axes,
    mode,
    ff_activation,
    expand_heads,
    fraction_to_rotate=0.25,
    conv_attention_kernel_width=None,
):
  """Returns a list of layers that implements a Transformer decoder block.
  The input to the block is a pair (activations, mask) where the mask encodes
  causal connections, preventing attention to future positions in the sequence.
  The block's outputs are the same type/shape as its inputs, so that multiple
  blocks can be chained together.
  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within decoder blocks. The same rate is also used
        for attention dropout in decoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.
  Returns:
    A list of layers that act in series as a (repeatable) decoder block.
  """
  def _CausalAttention():
    return RotaryCausalAttention(d_model, n_heads=n_heads, dropout=dropout,
                               kernel_width=conv_attention_kernel_width,
                               fraction_to_rotate=fraction_to_rotate,
                               mode=mode, expand_heads=expand_heads),

  def _FFBlock():
    return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                             ff_activation)

  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(
          tl.LayerNorm(),
          _CausalAttention(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          _FFBlock(),
          _Dropout(),
      ),
  ]


def _FeedForwardBlock(d_model,
                      d_ff,
                      dropout,
                      dropout_shared_axes,
                      mode,
                      activation):
  """Returns a list of layers that implements a feedforward block.
  Args:
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each block.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each block will include dropout; else, it will
        pass all values through unaltered.
    activation: Type of activation function at the end of each block; must
        be an activation-type subclass of :py:class:`Layer`.
  Returns:
    A list of layers that maps vectors to vectors.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Dense(d_ff),
      activation(),
      _Dropout(),
      tl.Dense(d_model),
  ]


@assert_shape('bld->bld')
def RotaryCausalAttention(
    d_model,
    fraction_to_rotate=0.25,
    kernel_width=None,
    n_heads=1,
    dropout=0.0,
    max_inference_length=2048,
    mode='train',
    expand_heads=False,
):
  """Returns a layer that maps activations to activations, with causal masking.
  Like :py:class:`Attention`, this layer type represents one pass of multi-head
  self-attention, but with causal masking rather than padding-based masking.
  Args:
    rotary_dim: Dimension of rotary embedding. If None, will default to d_head.
    d_model: Last/innermost dimension of activations in the input to and
        output from this layer.
    n_heads: Number of attention heads. Attention heads effectively split
        activation vectors into ``n_heads`` subvectors, of size
        ``d_feature / n_heads``.
    dropout: Probababilistic rate for attention dropout, which overrides
        (sets to zero) some attention strengths derived from query-key
        matching. As a result, on a given forward pass, some value vectors
        don't contribute to the output, analogous to how regular dropout can
        cause some node activations to be ignored. Applies only if layer is
        created in ``'train'`` mode.
    max_inference_length: Maximum sequence length allowed in non-training
        modes.
    mode: One of ``'train'``, ``'eval'``, or ``'predict'``.
  """
  if not expand_heads and d_model % n_heads != 0:
    raise ValueError(
        f'Dimensionality of feature embedding ({d_model}) is not a multiple '
        f'of the requested number of attention heads ({n_heads}).')

  def QKVLayer(layer_type: str):
    """Function returning the Q, K and V layer."""
    assert layer_type in ['Q', 'K', 'V'], 'Wrong layer type!'
    d_out = d_model * n_heads if expand_heads else d_model
    if kernel_width is not None and layer_type in ['Q', 'K']:
          # Dense + CausalDepthwiseConv should be equivalent to CausalConv
          # return cb.Serial(core.Dense(d_feature), convolution.CausalDepthwiseConv())
          return layers.CausalConv(d_out, kernel_width=kernel_width, mode=mode)
    return tl.Dense(d_out)

  return [
      tl.attention.ConfigurableAttention(
        QKVLayer(layer_type='Q'),
        QKVLayer(layer_type='K'),
        QKVLayer(layer_type='V'),
        tl.Dense(d_model),
        n_heads=n_heads,
        qkv_attention_layer=attention.DotProductCausalRotaryAttention(
          fraction_to_rotate=fraction_to_rotate,
          dropout=dropout,
          max_inference_length=max_inference_length,
          mode=mode)),
      *([tl.Dense(d_model)] if expand_heads else []),
  ]
