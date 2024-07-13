import math

from trax import fastmath
from trax import layers as tl
from trax.layers.assert_shape import assert_shape
from trax.fastmath import numpy as jnp


@assert_shape('bld,bld,bld->bld')
class DotProductCausalRotaryAttention(tl.DotProductCausalAttention):
    """Layer that applies rotary embedding and calls DotProductCausalAttention."""

    def __init__(self,
        fraction_to_rotate=0.25,
        dropout=0.0,
        max_inference_length=2048,
        mode='train'
    ):
        """Creates a :py:class:`DotProductCausalRotaryAttention` instance.

        For rotary embdding, see:
        * The original paper Su, et. al. RoFormer, https://arxiv.org/abs/2104.09864.
        * The blog post by the authors.
        * The blog post by EleutherAI https://blog.eleuther.ai/rotary-embeddings/.
        * Neel Nanda implementation: transformer_lens/components.py#L483.

        Args:
        rotary_dim: Dimension of rotary embedding. If not specified, defaults to d_head.
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
        super().__init__(dropout, max_inference_length, mode)
        self._fraction_to_rotate = fraction_to_rotate

    def forward(self, inputs):
        """Add rotary positional embeddings to inputs and call super.

        Args:
        inputs: A (queries, keys, values) tuple.
        """
        q, k, v = inputs  # [batch * heads, position, d_head]

        # Apply rotary embedding
        q, k = self.apply_rotary(q), self.apply_rotary(k)

        # Now that rotary embeddings have been added, call super.
        inputs = (q, k, v)
        return super().forward(inputs)

    def init_weights_and_state(self, input_signature):
        """Initializes this layer for fast inference, if in ``'predict'`` mode.

        See the perceiver implementation on the Google Research repo:
        perceiver-ar/blob/main/perceiver_ar/perceiver_ar_model.py
        """
        super().init_weights_and_state(input_signature)
        # input_signature[0].shape: [batch * heads, position, d_head]

        fraction_to_rotate = self._fraction_to_rotate
        if fraction_to_rotate > 1.0 or fraction_to_rotate <= 0.0:
            raise ValueError(
                f'fraction_to_rotate must be in (0, 1], got {fraction_to_rotate}.')

        def _to_even(x):
            return math.floor(x / 2.) * 2

        d_head = input_signature[0].shape[-1]
        self._rotary_dim = _to_even(d_head * fraction_to_rotate)

        # sin, cos  shape: [max_inference_length, rotary_dim]
        self._sin, self._cos = calculate_sin_cos_rotary(rotary_dim=self._rotary_dim,
            n_ctx=self._max_len)

    def apply_rotary(self, x):
        """Applies rotary embedding to x, which corresponds to k or q.
            x shape: [batch * heads, position, d_head]
        """
        rotary_dim = self._rotary_dim

        x_rot = x[..., :rotary_dim]
        x_unrotated = x[..., rotary_dim:]
        x_flip = rotate_every_two(x_rot)

        sin, cos = self._sin, self._cos  # each [n_ctx, d_heads]

        # In predict mode, you add just one new token.
        past_kv_pos_offset = 0
        if self._mode == 'predict':
            _, _, past_kv_pos_offset = self.state
        # In predict mode, take into account the number of previous tokens.
        x_rotated = (
            x_rot
            * fastmath.dynamic_slice_in_dim(cos, past_kv_pos_offset, x.shape[1], axis=0)
            + x_flip
            * fastmath.dynamic_slice_in_dim(sin, past_kv_pos_offset, x.shape[1], axis=0)
        )
        return jnp.concatenate([x_rotated, x_unrotated], axis=-1)


def calculate_sin_cos_rotary(rotary_dim: int, n_ctx: int, base: int=10000):
    pos = jnp.arange(n_ctx, dtype=jnp.float32)
    dim = jnp.arange(rotary_dim // 2, dtype=jnp.float32)
    # A set of frequencies evenly spaced in log space.
    # Each element of freq encodes one clock hand angular velocity.
    # Take the `base` to the power [0, 2/dim, ..., (dim-2)/dim].
    freq = base ** (dim / (rotary_dim / 2))
    # Repeat "pixels", so [f0, f1, ...] -> [f0, f0, f1, f1, ...]
    # Repetition needed to compute rotary embedding: x * cos + rotate(x) * sin
    # Same as einops.repeat(freq, 'd -> (d 2)')
    freq = jnp.repeat(freq, 2)
    # Create a n_ctx x rotary_dim tensor, where each column is an 
    # arithmetic sequence of angles in that frequency
    angles = pos[:, None] / freq[None, :]
    return jnp.sin(angles), jnp.cos(angles)


def rotate_every_two(x):
    """Splits x into blocks along the final axis and maps [x0, x1] to [-x1, x0]"""
    x0, x1 = x[..., ::2], x[..., 1::2]
    # Need to stack and rearrange, since DeviceArray is immutable.
    x = jnp.stack([-x1, x0], axis=-1)  # [batch * heads, position, d_head / 2, 2]
    # Rearrange to [batch * heads, position, d_head]
    # Equivalent to einops.rearrange(x, '... d j -> ... (d j)')
    return jnp.reshape(x, x.shape[:-2] + (-1,))
