import numpy as np
from trax import layers as tl
from trax import shapes

from .normalization import Normalizer


class TimeSeriesPredictor:
    """Training and prediction interface for time series modeling."""

    def __init__(
        self,
        model_body_fn,
        accelerate_predict_model,
        normalization,
        normalization_regularizer,
        context_type,
        input_vocab_sizes,
        decoder_fn,
    ):
        """Initializes TimeSeriesPredictor.

        Args:
            model_body_fn: Function mode -> model returning the body of a Trax
                sequence decoder model, e.g. code.models.TransformerBody, with
                input shape [batch_size, n_timesteps, d_in] and output shape
                [..., d_out].
            accelerate_predict_model (bool): Whether to jit the prediction model
                for acceleration. Should be turned off for testing.
            normalization (str): Normaliztion method. For possible options see
                predictors.normalization.Normalizer.
            normalization_regularizer (float): Normalization regularization
                term (constant added to the divisor).
            context_type (type): E.g. np.float32
            input_vocab_sizes (list[int]): Vocab sizes of the auxiliary input.
                If None, then it is skipped. Example: [30, 7, 24, 512], corresponds
                to a day in month, day in a week, hour, and time series id.
            decoder_fn: Function (model_body, **kwargs) -> decoder, where
                model_body is returned by model_body_fn.
        """
        self._model_body_fn = model_body_fn
        self._accelerate_predict_model = accelerate_predict_model
        self._predict_model = None

        self._normalizer: Normalizer = Normalizer.from_name(
            name=normalization,
            regularizer=normalization_regularizer,
        )

        self._n_inputs = len(input_vocab_sizes) if input_vocab_sizes is not None else 0

        self._init_state = None
        self._batch_size = None
        self._context_type = context_type
        self._decoder_fn = decoder_fn

    def init_state(self, batch_size):
        if self._init_state is None:
            self._predict_model = None
            (_, model_state) = self.predict_model.init(
                (
                    shapes.ShapeDtype((batch_size, 1), dtype=self._context_type),
                    shapes.ShapeDtype((batch_size, 1, self._n_inputs), dtype=np.int32),
                )
            )
            self._init_state = model_state
        elif batch_size != self._batch_size:
            raise ValueError(
                f"Model was initialized with batch size \
                {self._batch_size}, but a batch of size {batch_size} was \
                received."
            )
        self._batch_size = batch_size
        return self._init_state

    @property
    def predict_model(self):
        if self._predict_model is None:
            model = self.make_predict_model()
            if self._accelerate_predict_model:
                model = tl.Accelerate(model)
            self._predict_model = model
        return self._predict_model

    def make_train_eval_model(self, mode):
        """Returns a new model for training or evaluation."""
        raise NotImplementedError

    def make_predict_model(self):
        """Returns a new model for (autoregressive) prediction."""
        return self._decoder_fn(
            model_body=self._model_body_fn(mode="predict"), mode="predict"
        )

    def make_loss(self):
        """Returns a loss layer for model training."""
        raise NotImplementedError

    def before_training(self, inputs):
        """Called before training.

        Can be used to calculate some statistics on the dataset.

        Args:
            inputs: callable returning trax.inputs.Inputs.
        """
        pass

    def predict(self, weights, context, inputs, horizon_length):
        """Predicts a batch of time series up to a given horizon.

        Args:
            weights (pytree): Weights of the model.
            context (np.ndarray): Array of shape (batch_size, context_length)
                containing the past values of the time series.
            inputs (np.ndarray): Array of shape
                (batch_size, context_length + horizon_length, d_input)
                containing inputs (e.g. day of the week, time of day) for the
                entire sequence.
            horizon_length (int): Number of timesteps to predict.

        Returns:
            an array with the predicted series.
        """
        raise NotImplementedError

    def get_metrics(self):
        return {}

