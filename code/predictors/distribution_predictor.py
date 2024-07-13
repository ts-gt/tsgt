from functools import partial

import gin
import numpy as np
from trax import layers as tl
from trax.fastmath import numpy as jnp

import decoding
import distributions
from layers import Unsqueeze

from .inputs import InjectInputs
from .time_series_predictor import TimeSeriesPredictor


@gin.configurable(module="code.predictors")
class DistributionPredictor(TimeSeriesPredictor):
    """Training and prediction interface for time series modeling."""

    def __init__(
        self,
        model_body_fn=gin.REQUIRED,
        accelerate_predict_model=True,
        decoder_fn=None,
        d_in=256,
        input_vocab_sizes=None,
        normalization="per_ts",
        normalization_regularizer=1,
        distribution="gaussian",
    ):
        """Initializes DistributionPredictor.

        DistributionPredictor generates parameter(s) determining a
        distribution, from which one can sample prediction of
        a next time step of the time series.

        Args:
            model_body_fn: See TimeSeriesPredictor.__init__.
            accelerate_predict_model (bool): Whether to jit the model for
                acceleration.
            decoder_fn: Function (model_body, **kwargs) -> decoder, where
                model_body is returned by model_body_fn, mode is either 'train',
                'eval' or 'predict', and decoder is a simple wrapper for the
                model body adding a Dense layer for the outputs in the format of
                [mean, std]. If None, GaussianDecoder is used. Should be set
                only for testing.
            d_in (int): Depth of the input embedding.
            input_vocab_sizes (list[int]): Vocab sizes of the auxiliary input
                dimensions.
            normalization (str): Normaliztion method. For possible options see
                predictors.normalization.Normalizer.
            normalization_regularizer (float): Normalization regularization
                term (constant added to the divisor).
            distribution (str): Which distribution to model. See available options in
                distributions.py.
        """
        self._distribution: distributions.Distribution = (
            distributions.Distribution.from_name(distribution)
        )

        self.output_size = self._distribution.n_inputs

        if decoder_fn is None:
            decoder_fn = DistributionDecoder
        decoder_fn = partial(
            decoder_fn,
            d_emb=d_in,
            input_vocab_sizes=input_vocab_sizes,
            output_size=self.output_size,
        )

        super().__init__(
            model_body_fn=model_body_fn,
            accelerate_predict_model=accelerate_predict_model,
            normalization=normalization,
            normalization_regularizer=normalization_regularizer,
            context_type=np.float32,
            input_vocab_sizes=input_vocab_sizes,
            decoder_fn=decoder_fn,
        )

    def make_train_eval_model(self, mode):
        assert mode in ("train", "eval")
        decoder = self._decoder_fn(model_body=self._model_body_fn(mode=mode), mode=mode)
        use_mask = mode == "eval"  # in eval, normalize only based on 'seen' part of ts
        # Serial has SerialTraining function, but Gaussian doesn't have its own one,
        # so normalization layer is added here.
        return tl.Serial(
            self._normalizer.as_autoregressive_pipeline_layer(use_mask=use_mask),
            decoder,
        )

    def make_loss(self):
        return distributions.LogLoss(self._distribution)

    def predict(self, weights, context, inputs, horizon_length):
        (batch_size, _) = context.shape
        model_state = self.init_state(batch_size)
        self._predict_model.state = model_state

        # If weights are provided, they represent train_model. We extract decoder.
        self._predict_model.weights = weights and weights[1]

        norm_context, scaling_params, _ = self._normalizer.normalize(context)

        norm_pred = decoding.autoregressive_sample(
            self._predict_model,
            sample_fn=self._distribution.sample,
            context=norm_context,
            inputs=inputs,
            batch_size=batch_size,
            start_element=0,
            horizon_length=horizon_length,
        )
        norm_series = jnp.concatenate([norm_context, norm_pred], axis=-1)
        pred = self._normalizer.denormalize(norm_series, scaling_params)[
            :, -norm_pred.shape[-1] :
        ]
        return pred


def DistributionDecoder(
    model_body, d_emb, input_vocab_sizes, mode, output_size
):  # pylint: disable=invalid-name
    """Adds the input and output layers to a model body."""
    del mode
    return tl.Serial(
        # context, input
        # we don't need an embedding, but we need to add the embedding dimension
        Unsqueeze(),
        # context, input
        InjectInputs(input_vocab_sizes, d_emb),
        # context_and_input_emb
        model_body,
        # output
        tl.Dense(output_size),
        # [mean, std]
    )
