from typing import NamedTuple

import numpy as np
from trax.fastmath import numpy as jnp
from trax.layers.base import Fn


class Normalizer:
    def __init__(self, regularizer, apply):
        self.regularizer = regularizer
        self.apply = apply

    def normalize(self, data, mask=None):
        raise NotImplementedError()

    def denormalize(self, series, normalization_params):
        raise NotImplementedError()

    def as_autoregressive_pipeline_layer(self, use_mask: bool):
        """Assumes series == target and inputs is discarded by the model body. Used by SerialPredictor.

        Normalizes both series and target separately using exactly the same method.
        """
        assert use_mask in (True, False), "use_mask is not a boolean value"

        layer_name = f"Normalizer.normalize (use_mask={use_mask}, apply={self.apply})"

        def wrapper(series, inputs, target, mask):
            assert not use_mask or mask is not None, "incorrect mask"
            mask_ = mask if use_mask else None
            norm_series, _norm_params, _series_mask_modifier = self.normalize(
                data=series, mask=mask_
            )
            norm_target, _norm_params, target_mask_modifier = self.normalize(
                data=target, mask=mask_
            )
            return norm_series, inputs, norm_target, mask * target_mask_modifier

        return Fn(layer_name, wrapper, n_out=4)

    @staticmethod
    def from_name(name, regularizer) -> "Normalizer":
        if name == "per_ts":
            return PerTsNormalizer(regularizer=regularizer)
        if name == "per_batch":
            return PerBatchNormalizer(regularizer=regularizer)
        if name == "causal":
            return CausalNormalizer(regularizer=regularizer)
        if name is None or name == False:
            return NOPNormalizer(regularizer=regularizer)

        raise ValueError(f"Normalization method `{name}` does not exist!")


class PerTsNormalizer(Normalizer):
    class _NormalizationParams(NamedTuple):
        scaling_factor: np.ndarray

    def __init__(self, regularizer):
        super(PerTsNormalizer, self).__init__(regularizer, "per_ts")

    def normalize(self, data, mask=None):
        # TODO what does mask actually define, which values to use to calcualte mean or
        # over which values to  calculate gradients
        assert mask is None or data.shape == mask.shape

        # series shape (batch, length)

        if mask is None:
            use_for_statistics = jnp.ones_like(data)
        else:
            use_for_statistics = mask

        absmean = jnp.abs(data).mean(
            axis=1, where=(use_for_statistics > 0), keepdims=True
        )

        scaling_factor = absmean + self.regularizer
        scaled = jnp.divide(data, scaling_factor)

        return scaled, self._NormalizationParams(scaling_factor), jnp.ones_like(scaled)

    def denormalize(self, series, normalization_params: _NormalizationParams):
        scaling_factor = normalization_params.scaling_factor

        assert jnp.squeeze(scaling_factor, axis=1).shape == (series.shape[0],)
        scaled = jnp.multiply(series, scaling_factor)

        return scaled


class PerBatchNormalizer(Normalizer):
    class _NormalizationParams(NamedTuple):
        scaling_factor: np.ndarray

    def __init__(self, regularizer):
        super(PerBatchNormalizer, self).__init__(regularizer, "per_batch")

    def normalize(self, data, mask=None):
        assert mask is None or data.shape == mask.shape

        # series shape (batch, length)

        if mask is None:
            use_for_statistics = jnp.ones_like(data)
        else:
            use_for_statistics = 1 - mask

        absmean = jnp.abs(data).mean(where=(use_for_statistics > 0), keepdims=True)

        scaling_factor = absmean + self.regularizer
        scaled = jnp.divide(data, scaling_factor)

        return scaled, self._NormalizationParams(scaling_factor), jnp.ones_like(scaled)

    def denormalize(self, series, normalization_params):
        scaling_factor = normalization_params.scaling_factor

        assert jnp.squeeze(scaling_factor).shape == ()
        scaled = jnp.multiply(series, scaling_factor)

        return scaled


class CausalNormalizer(Normalizer):
    class _NormalizationParams(NamedTuple):
        first_value: np.ndarray

    def __init__(self, regularizer):
        super(CausalNormalizer, self).__init__(regularizer, "causal")

    def normalize(self, data, mask=None):
        CAUSAL_NORMALIZATION_GRADIENT_STOP_INDEX = 10
        assert mask is None or data.shape == mask.shape

        # series shape (batch, length)

        # We do not use mask since causal normalization prevents data leak.
        abssum = jnp.cumsum(jnp.abs(data), axis=-1)
        absmean = abssum / (jnp.arange(data.shape[-1]) + 1)

        scaling_factor = absmean + self.regularizer
        scaled = jnp.divide(data, scaling_factor)

        params = self._NormalizationParams(first_value=data[:, 0])

        mask_diff = np.ones(scaled.shape)
        mask_diff[:, :CAUSAL_NORMALIZATION_GRADIENT_STOP_INDEX] = 0

        return scaled, params, mask_diff

    def denormalize(self, series, normalization_params):
        first_value = normalization_params.first_value

        prefix_sum = np.abs(first_value)
        scaled = np.empty_like(series)
        for n in range(series.shape[1]):
            if n == 0:
                scaled[:, n] = first_value
            else:
                predicted_value = series[:, n]
                true_value = (
                    predicted_value * (prefix_sum + self.regularizer * (n + 1))
                ) / (n + 1 - np.abs(predicted_value))
                prefix_sum += np.abs(true_value)
                scaled[:, n] = true_value

        return scaled


class NOPNormalizer(Normalizer):
    def __init__(self, regularizer):
        super(NOPNormalizer, self).__init__(regularizer, "NOP")

    def normalize(self, data, mask=None):
        return (data, None, jnp.ones_like(data))

    def denormalize(self, series, normalization_params):
        return series
