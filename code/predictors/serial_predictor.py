from functools import partial

import gin
import gym
import numpy as np
from trax import layers as tl
from trax import shapes
from trax.fastmath import numpy as jnp
from trax.layers.base import Fn
from trax.rl import serialization_utils as srl_utils

import decoding
import distributions
import serializers
from layers import Unsqueeze
from metrics import WeightedSmoothedCategoryCrossEntropy

from .inputs import InjectInputs
from .normalization import Normalizer
from .time_series_predictor import TimeSeriesPredictor


@gin.configurable(module="code.predictors")
class SerialPredictor(TimeSeriesPredictor):
    """Time series predictor based on serialization."""

    def __init__(
        self,
        model_body_fn=gin.REQUIRED,
        decoder_fn=None,
        d_in=256,
        vocab_size=64,
        precision=2,
        significance_decay=0.7,
        low=0.0,
        high=1.0,
        accelerate_predict_model=True,
        input_vocab_sizes=None,
        normalization="per_ts",
        normalization_regularizer=1,
        label_smoothing=None,
        first_digit_mode="uniform",
        clip_or_squash="clip",
    ):
        """Initializes SerialPredictor.

        SerialPredictor is based on serialization: turning the time series into
        a sequence of discrete symbols, which can be effectively modeled using
        e.g. the Transformer language model.

        Args:
            model_body_fn: Function (mode, precision) -> model returning the body of a Trax
                sequence decoder model, e.g. code.models.TransformerBody, with
                input shape [batch_size, n_timesteps, d_in] and output shape
                [..., d_out].
            decoder_fn: Function (model_body, d_emb, mode) -> decoder, where
                model_body is returned by model_body_fn, d_emb is the depth of
                the symbol embedding, mode is either 'train', 'eval' or
                'predict', and decoder is a discrete sequence decoder model. If
                None, SerialDecoder is used. Should be set only for testing.
            d_in (int): Depth of the symbol embedding.
            vocab_size (int): Vocabulary size (number of distinct symbols).
            precision (int): Number of symbols used to encode a single float.
            significance_decay (float): Decay factor for exponential weighting
                of the symbolwise loss.
            low (float): Minimum representable value.
            high (float): Maximum representable value.
            accelerate_predict_model (bool): Whether to jit the model for
                acceleration.
            input_vocab_sizes (list[int]): Vocab sizes of the auxiliary input.
                If None, then it is skipped. Example: [30, 7, 24, 512], corresponds
                to a day in month, day in a week, hour, and time series id.
            normalization (bool): Whether to normalize the series.
            normalization_regularizer (float): Normalization regularization
                term (constant added to the divisor).
            label_smoothing (float): If it is not equal to None, then labels are
                smoothed (discretized normal distribution std=label_smoothing)
            first_digit_mode (str): How to encode the first digit. The available
                modes are 'uniform' and 'quantile'.
            clip_or_squash (str): Either 'clip' or 'squash', decides on serialization
                strategy.
        """
        self._serializer = serializers.BoxSpaceSerializer(
            space=gym.spaces.Box(shape=(), low=low, high=high),
            vocab_size=vocab_size,
            precision=precision,
            first_digit_mode=first_digit_mode,
            clip_or_squash=clip_or_squash,
        )

        if decoder_fn is None:
            decoder_fn = SerialDecoder
        decoder_fn = partial(
            decoder_fn,
            serializer=self._serializer,
            d_emb=d_in,
            input_vocab_sizes=input_vocab_sizes,
        )

        self._d_in = d_in
        self._precision = precision
        self._significance_decay = significance_decay
        self._label_smoothing = label_smoothing
        self._categorical = distributions.Categorical(n_categories=vocab_size)

        super().__init__(
            model_body_fn=partial(model_body_fn, precision=precision),
            accelerate_predict_model=accelerate_predict_model,
            normalization=normalization,
            normalization_regularizer=normalization_regularizer,
            context_type=np.int32,
            input_vocab_sizes=input_vocab_sizes,
            decoder_fn=decoder_fn,
        )

    def make_train_eval_model(self, mode):
        assert mode in ("train", "eval")
        use_mask = mode == "eval"  # in eval, normalize only based on 'seen' part of ts
        return SerialTraining(
            decoder_model=self._decoder_fn(
                model_body=self._model_body_fn(mode=mode),
                mode=mode,
            ),
            serializer=self._serializer,
            significance_decay=self._significance_decay,
            use_mask=use_mask,
            normalizer=self._normalizer,
        )

    def make_loss(self):
        if self._label_smoothing is None:
            return tl.WeightedCategoryCrossEntropy()
        return WeightedSmoothedCategoryCrossEntropy(
            self._label_smoothing, self._precision
        )

    def before_training(self, inputs):
        # in case of serialzer._first_digit_mode
        def normalize_one(inp):
            series, _, _, _ = inp
            return self._normalizer.normalize(series)[0]

        input_stream = map(normalize_one, inputs().train_stream(None))
        self._serializer.fit(input_stream)

    def predict(self, weights, context, inputs, horizon_length):
        (batch_size, _) = context.shape
        # The following line "resets" `_predict_model`'s state.
        # TODO: explain why?
        self._predict_model.state = self.init_state(batch_size)

        # If weights are provided, they represent train_model. We extract decoder.
        # TODO: explain.
        self._predict_model.weights = weights and weights[2]

        norm_context, scaling_params, _ = self._normalizer.normalize(context)
        context_repr = self._serializer.serialize(norm_context)
        repr_len = self._serializer.representation_length

        # Upsample the inputs (along time axis) to match the serialized sequence length.
        inputs = np.repeat(inputs, repeats=repr_len, axis=1)

        pred_repr = decoding.autoregressive_sample(
            self._predict_model,
            sample_fn=self._categorical.sample,
            context=context_repr,
            inputs=inputs,
            batch_size=batch_size,
            start_element=0,
            horizon_length=repr_len * horizon_length,
        )

        pred_repr = np.reshape(pred_repr, (-1, repr_len))
        norm_pred = self._serializer.deserialize(pred_repr)
        norm_pred = np.reshape(norm_pred, (batch_size, -1))
        norm_series = jnp.concatenate([norm_context, norm_pred], axis=-1)
        pred = self._normalizer.denormalize(norm_series, scaling_params)[
            :, -norm_pred.shape[-1] :
        ]
        return pred


def SerialDecoder(model_body, serializer, d_emb, input_vocab_sizes, mode):
    """Adds discrete input and output layers to a model body."""
    del mode
    return tl.Serial(
        # context_repr, inputs
        tl.Embedding(serializer.vocab_size, d_emb),
        # context_emb, inputs
        InjectInputs(input_vocab_sizes, d_emb),
        # context_and_input_emb
        model_body,
        # output
        tl.Dense(serializer.vocab_size),
        # output_logits
    )


def SerialTraining(
    decoder_model,
    serializer,
    significance_decay,
    use_mask,
    normalizer: Normalizer,
):  # pylint: disable=invalid-name
    """Wraps a sequence decoder in serialization machinery for training."""
    weigh_by_significance = [
        # mask
        srl_utils.RepresentationMask(serializer),
        # repr_mask
        srl_utils.SignificanceWeights(serializer=serializer, decay=significance_decay),
        # weights
        tl.Flatten(),
    ]
    serialize = [
        srl_utils.Serialize(serializer),
        tl.Flatten(),
    ]

    def upsample(inputs):
        """Upsample the inputs to match the serialized sequence length."""
        return jnp.repeat(
            inputs,
            repeats=serializer.representation_length,
            axis=1,  # Time axis.
        )

    return tl.Serial(
        normalizer.as_autoregressive_pipeline_layer(
            use_mask
        ),  # train mode, so use mask
        # (context, input, target, mask)
        # tl.Parallel()
        tl.Parallel(serialize, tl.Fn("Upsample", upsample), serialize),
        # (context_repr, input, target_repr, mask)
        decoder_model,
        # (output_logits, target_repr, mask)
        tl.Parallel(None, None, weigh_by_significance),
        # (output_logits, target_repr, weights)
    )
