from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers.base import Fn
from trax.layers import core


def _smooth_target(targets, n_categories, precision, std):
    stds = jnp.ones(targets.shape[-1])
    for i in range(precision):
        stds = stds.at[i::precision].set(i)
    stds = jnp.expand_dims(std * n_categories ** stds, axis=-1)

    target_smooth = (jnp.zeros(targets.shape + (n_categories,))
        + jnp.arange(n_categories)
        - jnp.expand_dims(targets, axis=-1)) / stds

    log_pdf = -target_smooth ** 2 / 2
    normalizer = fastmath.logsumexp(log_pdf, axis=-1, keepdims=True)
    return jnp.exp(log_pdf - normalizer)


def WeightedSmoothedCategoryCrossEntropy(std, precision):
    """Computes category cross entropy with labels smoothed using
    discretized normal distribution."""

    def f(model_output, targets, weights):
        n_categories = model_output.shape[-1]
        target_smooth = _smooth_target(targets, n_categories, precision, std)
        model_log_distributions = core.log_softmax(model_output)
        cross_entropies = - jnp.sum(target_smooth * model_log_distributions, axis=-1)
        return (jnp.sum(cross_entropies * weights) / jnp.sum(weights))

    return Fn('WeightedSmoothedCategoryCrossEntropy', f)
