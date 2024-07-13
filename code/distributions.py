"""Probability distributions for RL training in Trax."""

import gin
import gym
import numpy as np
from trax import layers as tl
from trax.fastmath import numpy as jnp


class Distribution:
    """Abstract class for parametrized probability distributions."""

    @property
    def n_inputs(self):
        """Returns the number of inputs to the distribution (i.e. parameters)."""
        raise NotImplementedError

    def sample(self, inputs, temperature=1.0):
        """Samples a point from the distribution.

        Args:
            inputs (jnp.ndarray): Distribution inputs. Shape is subclass-specific.
                Broadcasts along the first dimensions. For example, in the categorical
                distribution parameter shape is (C,), where C is the number of
                categories. If (B, C) is passed, the object will represent a batch of B
                categorical distributions with different parameters.
            temperature: sampling temperature; 1.0 is default, at 0.0 chooses
                the most probable (preferred) action.

        Returns:
            Sampled point of shape dependent on the subclass and on the shape of
            inputs.
        """
        raise NotImplementedError

    def log_prob(self, inputs, point):
        """Retrieves log probability (or log probability density) of a point.

        Args:
            inputs (jnp.ndarray): Distribution parameters.
            point (jnp.ndarray): Point from the distribution. Shape should be
                consistent with inputs.

        Returns:
            Array of log probabilities of points in the distribution.
        """
        raise NotImplementedError

    def LogProb(self):  # pylint: disable=invalid-name
        """Builds a log probability layer for this distribution."""
        return tl.Fn(
            "LogProb", lambda inputs, point: self.log_prob(inputs, point)
        )  # pylint: disable=unnecessary-lambda

    @staticmethod
    def from_name(name, **kwargs) -> "Distribution":
        """Factory function for distibutions. Returns instance of distribution specified
        by name."""
        if name == "categorical":
            return Categorical(**kwargs)
        if name == "gaussian":
            return Gaussian(**kwargs)
        if name == "laplace":
            return Laplace(**kwargs)
        if name == "cauchy":
            return Cauchy(**kwargs)
        if name == "tstudent":
            return TStudent(**kwargs)

        raise ValueError(f"Distribution {name} does not exist!")


@gin.configurable(module="code.distributions", denylist=["n_categories", "shape"])
class Categorical(Distribution):
    """Categorical distribution parametrized by logits."""

    def __init__(self, n_categories, shape=()):
        """Initializes Categorical distribution.

        Args:
            n_categories (int): Number of categories.
            shape (tuple): Shape of the sample.
        """
        self._n_categories = n_categories
        self._shape = shape

    @property
    def n_inputs(self):
        return np.prod(self._shape, dtype=jnp.int32) * self._n_categories

    def _unflatten_inputs(self, inputs):
        return jnp.reshape(
            inputs, inputs.shape[:-1] + self._shape + (self._n_categories,)
        )

    def sample(self, inputs, temperature=1.0):
        # No need for LogSoftmax with sampling - softmax normalization is
        # subtracting a constant from every logit, and sampling is taking
        # a max over logits plus noise, so invariant to adding a constant.
        if temperature == 0.0:
            return jnp.argmax(self._unflatten_inputs(inputs), axis=-1)
        return tl.logsoftmax_sample(self._unflatten_inputs(inputs), temperature)

    def log_prob(self, inputs, point):
        inputs = tl.LogSoftmax()(self._unflatten_inputs(inputs))
        return jnp.sum(
            # Select the logits specified by point.
            inputs * tl.one_hot(point, self._n_categories),
            # Sum over the parameter dimensions.
            axis=[-a for a in range(1, len(self._shape) + 2)],
        )

    def entropy(self, inputs):
        log_probs = tl.LogSoftmax()(inputs)
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs, axis=-1)


class ScaleLocationDistribution(Distribution):
    """Independent multivariate distribution parametrized by location and scale."""

    def __init__(self, shape, scale, learn_scale):
        """Initializes the distribution.

        Args:
            shape (tuple): Shape of the sample.
            scale (float): E.g. standard deviation, shared across the whole sample.
            learn_scale (str or None): How to learn the scale - 'shared'
                to have a single, shared scale parameter, or 'separate' to have separate
                parameters for each dimension.
            sampling_function (callable): Used to sample from distibution.
        """
        self._shape = shape
        self._scale = scale
        self._learn_scale = learn_scale

    def sampling_function(self, location, scale, temperature):
        raise NotImplementedError()

    @property
    def _n_dims(self):
        return np.prod(self._shape, dtype=jnp.int32)

    def _params(self, inputs):
        """Extracts the location and scale parameters from the inputs."""
        if inputs.shape[-1] != self.n_inputs:
            raise ValueError(
                f"Invalid distribution parametrization - expected {self.n_inputs} "
                f"parameters, got {inputs.shape[-1]}. Input shape: {inputs.shape}."
            )
        n_dims = self._n_dims
        # Split the distribution inputs into two parts: mean and std.
        location = inputs[..., :n_dims]
        if self._learn_scale is not None:
            scale = inputs[..., n_dims:]
            # Std should be non-negative - but it might not be when it comes
            # from a network output. Therefore we will use Softplus on it.
            scale = tl.Softplus()(scale) + self._scale
        else:
            scale = self._scale
        # In case of constant or shared std, upsample it to the same dimensionality
        # as the means.
        scale = jnp.broadcast_to(scale, location.shape)
        return (location, scale)

    @property
    def n_inputs(self):
        n_dims = self._n_dims
        return {
            None: n_dims,
            "shared": n_dims + 1,
            "separate": n_dims * 2,
        }[self._learn_scale]

    def sample(self, inputs, temperature=1.0):
        (location, scale) = self._params(inputs)
        location = jnp.reshape(location, location.shape[:-1] + self._shape)
        scale = jnp.reshape(scale, scale.shape[:-1] + self._shape)
        if temperature == 0:
            return location
            # this seemingly strange if solves the problem
            # of calling np/jnp.random in the metric PreferredMove

        return self.sampling_function(location, scale, temperature)


@gin.configurable(module="code.distributions", denylist=["shape"])
class Gaussian(ScaleLocationDistribution):
    """Independent multivariate Gaussian distribution."""

    def __init__(self, shape=(), scale=1.0, learn_scale=None):
        super().__init__(shape, scale, learn_scale)

    def sampling_function(self, location, scale, temperature):
        return np.random.normal(loc=location, scale=(scale * temperature))

    def log_prob(self, inputs, point):
        point = point.reshape(inputs.shape[:-1] + (-1,))
        (mean, std) = self._params(inputs)
        return -jnp.sum(
            # Scaled distance.
            (point - mean) ** 2 / (2 * std**2) +
            # Normalizing constant.
            (jnp.log(std) + jnp.log(jnp.sqrt(2 * jnp.pi))),
            axis=-1,
        )

    def entropy(self, inputs):
        (_, std) = self._params(inputs)
        return jnp.sum(jnp.exp(std) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


@gin.configurable(module="code.distributions", denylist=["shape"])
class Laplace(ScaleLocationDistribution):
    """Independent multivariate Laplace distribution."""

    def __init__(self, shape=(), scale=1.0, learn_scale=None):
        super().__init__(shape, scale, learn_scale)

    def sampling_function(self, location, scale, temperature):
        return np.random.laplace(loc=location, scale=(scale * temperature))

    def log_prob(self, inputs, point):
        point = point.reshape(inputs.shape[:-1] + (-1,))
        location, diversity = self._params(inputs)
        return -jnp.sum(
            # Scaled distance.
            jnp.abs(point - location) / diversity +
            # Normalizing constant.
            jnp.log(2 * diversity),
            axis=-1,
        )

    def entropy(self, inputs):
        (_, diversity) = self._params(inputs)
        return jnp.sum(jnp.log(2.0 * diversity * jnp.e), axis=-1)


@gin.configurable(module="code.distributions", denylist=["shape"])
class Cauchy(ScaleLocationDistribution):
    """Independent multivariate Cauchy distribution."""

    def __init__(self, shape=(), scale=1.0, learn_scale=None):
        super().__init__(shape, scale, learn_scale)

    def sampling_function(self, location, scale, temperature):
        return (
            np.random.standard_cauchy(size=location.shape) * scale * temperature
            + location
        )

    def log_prob(self, inputs, point):
        point = point.reshape(inputs.shape[:-1] + (-1,))
        location, scale = self._params(inputs)
        return -jnp.sum(
            # Scaled distance.
            jnp.log((point - location) ** 2 + scale**2) +
            # Normalizing constant.
            jnp.log(jnp.pi / scale),
            axis=-1,
        )

    def entropy(self, inputs):
        (_, scale) = self._params(inputs)
        return jnp.sum(jnp.log(4.0 * scale * jnp.pi), axis=-1)


@gin.configurable(module="code.distributions", denylist=["shape"])
class TStudent(ScaleLocationDistribution):
    """Independent multivariate T-student distribution with 5 degrees of freedom.
    Parametrized by location and scale.

    WARNING: it does not implement proper log_prob function (does not sum up to 0),
    only an equivalent from the perspective of gradient optimization.
    """

    def __init__(self, shape=(), scale=1.0, learn_scale=None):
        super().__init__(shape, scale, learn_scale)
        self.degree = 5

    def sampling_function(self, location, scale, temperature):
        return (
            np.random.standard_t(5, size=location.shape) * scale * temperature
            + location
        )

    def log_prob(self, inputs, point):
        point = point.reshape(inputs.shape[:-1] + (-1,))
        location, scale = self._params(inputs)
        return jnp.sum(
            jnp.log(1 + (point - location) ** 2 / scale**2 / self.degree)
            * (-(self.degree + 1) / 2)
            - jnp.log(scale),
            axis=-1,
        )


def create_distribution(space):
    """Creates a Distribution for the given Gym space."""
    if isinstance(space, gym.spaces.Discrete):
        return Categorical(shape=(), n_categories=space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        assert space.nvec.size
        assert min(space.nvec) == max(
            space.nvec
        ), f"Every dimension must have the same number of categories, got {space.nvec}."
        return Categorical(shape=(len(space.nvec),), n_categories=space.nvec[0])
    elif isinstance(space, gym.spaces.Box):
        return Gaussian(shape=space.shape)
    else:
        raise TypeError("Space {} unavailable as a distribution support.")


def LogLoss(distribution: Distribution):  # pylint: disable=invalid-name
    """Builds a log loss layer for a Distribution."""
    return tl.Serial(
        distribution.LogProb(), tl.Negate(), tl.WeightedSum(), name="LogLoss"
    )
