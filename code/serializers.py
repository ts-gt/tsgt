"""Serialization of the elements of Gym spaces into discrete sequences."""

import copy

from absl import logging
import gin
import gym
from jax import numpy as np
from sklearn import preprocessing as skp
from trax.rl import space_serializer


@gin.configurable(denylist=["space", "vocab_size"], module="code.serializers")
class BoxSpaceSerializer(space_serializer.SpaceSerializer):
    """Serializer for gym.spaces.Box.

    Assumes that the space is bounded. Internally rescales it to the [0, 1]
    interval and uses a fixed-precision encoding.
    """

    space_type = gym.spaces.Box

    def __init__(
        self,
        space,
        vocab_size,
        precision=2,
        max_range=(-100.0, 100.0),
        first_digit_mode="uniform",
        quantile_fit_n_points=(1024**2),
        clip_or_squash="clip",
    ):
        """Initializes BoxSpaceSerializer.

        Args:
            space: Gym space.
            vocab_size (int): Number of distinct tokens.
            precision (int): Number of tokens to use to encode each number.
            max_range (tuple[float, float]): Maximum representable range.
            first_digit_mode (str): How to encode the first digit. The available
                modes are 'uniform' and 'quantile'.
            quantile_fit_n_points (int): Number of datapoints (total timesteps)
                to fit the histogram on, for first_digit_mode == 'quantile'.
        """
        self._precision = precision
        self._first_digit_mode = first_digit_mode
        self._quantile_fit_n_points = quantile_fit_n_points

        assert space.shape == ()

        # Some gym envs (e.g. CartPole) have unreasonably high bounds for
        # observations. We clip so we can represent them.
        bounded_space = copy.copy(space)
        (min_low, max_high) = max_range
        bounded_space.low = np.maximum(space.low, min_low)
        bounded_space.high = np.minimum(space.high, max_high)
        if not np.allclose(bounded_space.low, space.low) or not np.allclose(
            bounded_space.high, space.high
        ):
            logging.warning(
                "Space limits %s, %s out of bounds %s. Clipping to %s, %s.",
                str(space.low),
                str(space.high),
                str(max_range),
                str(bounded_space.low),
                str(bounded_space.high),
            )

        self._discretizer = None

        super().__init__(bounded_space, vocab_size)
        assert clip_or_squash in ("clip", "squash")
        self.clip_or_squash = clip_or_squash

    def _preprocess(self, data):
        array = np.array(data)
        array = (array - self._space.low) / (
            self._space.high - self._space.low
        )  # move to [0, 1]
        if self.clip_or_squash == "clip":
            # move to [0, 1] than clip
            array = np.clip(array, 0, 1)
        elif self.clip_or_squash == "squash":
            # if self._space.low >= 0:
            #     # move to [0, 5] than apply sigmoid (gets [0.5, 1]) then move to [0, 1]
            # else:
            #     # move to [-5, 5] than apply sigmoid
            array = array * (5 + 5 * (self._space.low < 0)) - 5 * (self._space.low < 0)
            array = sigmoid(array)
            array = array * (self._space.low < 0) + (array - 0.5) * 2 * (
                self._space.low >= 0
            )
        return array

    def _postprocess(self, data):
        array = np.asarray(data)
        if self.clip_or_squash == "squash":
            array = (array / 2 + 0.5) * (self._space.low >= 0) + array * (
                self._space.low < 0
            )
            array = inv_sigmoid(array)  # gets ~ [0, 5]
            array = array / (5 + 5 * (self._space.low < 0)) + 0.5 * (
                self._space.low < 0
            )

        array = (
            array * (self._space.high - self._space.low) + self._space.low
        )  # from [0, 1] move back to original space
        return array

    def fit(self, input_stream):
        """Fits the serializer on the data, if necessary.

        Args:
            input_stream: Generator of arrays of shape (batch_size, length).
        """
        if self._first_digit_mode != "quantile":
            return

        n_points_left = self._quantile_fit_n_points
        point_batches = []
        while n_points_left > 0:
            series = next(input_stream)
            series = self._preprocess(series)
            point_batches.append(np.reshape(series, -1))
            n_points_left -= series.size
        points = np.concatenate(point_batches)
        points = np.reshape(points, (-1, 1))

        self._discretizer = skp.KBinsDiscretizer(
            strategy="quantile", encode="ordinal", n_bins=self._vocab_size
        )
        self._discretizer.fit(points)
        self._discretizer.bin_edges_[0][0] = 0.0
        self._discretizer.bin_edges_[0][-1] = 1.0

    def serialize(self, data):
        array = self._preprocess(data)
        batch_size = array.shape[0]
        digits = []
        current_range = 1
        for digit_index in range(-1, -self._precision - 1, -1):
            if digit_index == -1 and self._first_digit_mode == "quantile":
                edges = np.array(self._discretizer.bin_edges_[0])
                # We can't use self._discretizer.transform because it doesn't
                # support JAX.
                digit = np.searchsorted(edges[1:-1], array, side="right")
                lower_bound = edges[digit]
                current_range = edges[digit + 1] - edges[digit]
            else:
                threshold = current_range / self._vocab_size
                digit = np.array(array / threshold).astype(np.int32)
                # For the corner case of x ~= high.
                digit = np.where(digit == self._vocab_size, digit - 1, digit)
                lower_bound = digit * threshold
                current_range /= self._vocab_size

            digits.append(digit)
            array -= lower_bound

        digits = np.stack(digits, axis=-1)
        return np.reshape(digits, (batch_size, -1))

    def deserialize(self, representation):
        digits = np.array(representation)
        batch_size = digits.shape[0]
        digits = np.reshape(digits, (batch_size, self._precision))
        array = np.zeros(digits.shape[:-1])
        current_range = 1
        for digit_index_in_seq in range(self._precision):
            if digit_index_in_seq == 0 and self._first_digit_mode == "quantile":
                digit = digits[..., 0]
                edges = np.array(self._discretizer.bin_edges_[0])
                array = edges[digit]
                current_range = edges[digit + 1] - edges[digit]
            else:
                digit_index = -digit_index_in_seq - 1
                array += (
                    current_range / self._vocab_size * digits[..., digit_index_in_seq]
                )
                current_range /= self._vocab_size
        array = np.reshape(array, (batch_size,) + self._space.shape)
        return self._postprocess(array)

    @property
    def representation_length(self):
        return self._precision * self._space.low.size

    @property
    def significance_map(self):
        return np.reshape(
            np.broadcast_to(
                np.arange(self._precision),
                self._space.shape + (self._precision,),
            ),
            -1,
        )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(s):
    return np.log(s / (1 - s))
