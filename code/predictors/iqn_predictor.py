from functools import partial

import decoding
import distributions
import gin
import numpy as np
from layers import Unsqueeze
from trax import layers as tl
from trax.fastmath import numpy as jnp

from .inputs import InjectInputs
from .time_series_predictor import TimeSeriesPredictor


@gin.configurable(module="code.predictors")
class IQNPredictor(TimeSeriesPredictor):
    """Training and prediction interface for time series modeling.
    Uses Implicit Quantile Network idea presented in
    'Probabilistic Time Series Forecasting with Implicit Quantile Networks'"""

    def __init__(
        self,
        model_body_fn=gin.REQUIRED,
        accelerate_predict_model=True,
        decoder_fn=None,
        d_in=256,
        input_vocab_sizes=None,
        normalization="per_ts",
        normalization_regularizer=1,
    ):
        """Initializes IQNPredictor.

        IQNPredictor generates parameter(s) determining a
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
        self.iqm = None
        self._iqm_fn = partial(ImplicitQuantileModule, d_emb=d_in)
        if decoder_fn is None:
            decoder_fn = IQNDecoder
        decoder_fn = partial(
            decoder_fn, d_emb=d_in, input_vocab_sizes=input_vocab_sizes, iqm=self.iqm
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
        self.iqm = self._iqm_fn()
        use_mask = mode == "eval"  # in eval, normalize only based on 'seen' part of ts
        # Serial has SerialTraining function, but Gaussian doesn't have its own one,
        # so normalization layer is added here.
        return tl.Serial(
            self._normalizer.as_autoregressive_pipeline_layer(use_mask=use_mask),
            decoder,
            self.iqm,
        )

    def make_loss(self):
        return QuantileLoss()

    def predict(self, weights, context, inputs, horizon_length):
        (batch_size, _) = context.shape
        model_state = self.init_state(batch_size)
        self._predict_model.state = model_state

        # If weights are provided, they represent train_model. We extract decoder.
        self._predict_model.weights = weights and weights[1]
        norm_context, scaling_params, _ = self._normalizer.normalize(context)

        sample_fn = tl.Serial(
            tl.Fn(
                "extend_dim", lambda x: jnp.expand_dims(x, axis=1)
            ),  # add missing time axis
            self.iqm,
            tl.Parallel(None, tl.Drop()),  # take first output of iqm
            tl.Fn("squeeze", lambda x: jnp.squeeze(x, axis=1)),  # remove time axis
        )

        norm_pred = decoding.autoregressive_sample(
            self._predict_model,
            sample_fn=sample_fn,
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


def QuantileLoss() -> tl.Layer:
    def q_loss(quantile_forecast, tau, target):
        print("QL Input shapes:", quantile_forecast.shape, tau.shape, target.shape)
        return jnp.abs(
            (quantile_forecast - target)
            * ((target <= quantile_forecast).astype("float") - tau)
        )

    return tl.Serial(
        tl.Fn("quantile_loss", q_loss), tl.WeightedSum(), name="QuantileLoss"
    )


def QuantileLayer(d_emb) -> tl.Layer:
    "Takes tau and returns its embedding"
    n_cos_embedding = 64

    def cos_embed(tau):
        # input shape: (batch, series_len)
        # output shape: (batch, series_len, d_emb)
        integers = jnp.expand_dims(jnp.ones_like(tau), axis=-1)
        integers = integers * jnp.expand_dims(
            jnp.arange(0, n_cos_embedding), axis=(0, 1)
        )
        # shape (batch, series_len, n_cos_embedding)
        result = jnp.cos(jnp.pi * jnp.expand_dims(tau, axis=-1) * integers)
        return result

    return tl.Serial(
        # tau,
        tl.Fn("cos_embed", cos_embed),
        # emb_tau,
        tl.Dense(n_cos_embedding),
        tl.LeakyRelu(a=0.1),  # Originally `a`` was trainable
        tl.Dense(d_emb),
    )


def ImplicitQuantileModule(d_emb: int) -> tl.Layer:
    """Computes quantile balue based on ts embedding and tau,
    returns quantile value and tau"""

    def sample_tau(x):
        # x shape: (batch, series_len, d_emb)
        return np.random.uniform(size=x.shape[:-1])

    def combine_embeddings(tau_emb, model_output):
        return model_output * (jnp.ones_like(tau_emb) + tau_emb)

    output_layer = tl.Serial(
        tl.Dense(d_emb),
        tl.Softplus(),
        tl.Dense(1),
        tl.Fn("squeeze", lambda x: jnp.squeeze(x, axis=-1)),
    )

    return tl.Serial(
        # output
        tl.Branch(tl.Fn("sample_tau", sample_tau), None),
        # tau, output
        tl.Dup(),
        # tau, tau, output
        tl.Parallel(None, tl.Swap()),
        # tau, output, tau
        QuantileLayer(d_emb),
        # tau_emb, output, tau
        tl.Fn("combine_embeddings", combine_embeddings),
        # new_input, tau
        output_layer,
        # qantile_value, tau
    )


def IQNDecoder(
    model_body, d_emb, input_vocab_sizes, mode, iqm
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
        # iqm,
        # quantile_value, tau
    )
