import os
from typing import Callable, Tuple

import gin
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


@gin.configurable(module="code.datasets")
def load_csv(csv_dataset_path, resample_freq=None):
    def _thunk():
        folder = "/dataset" if os.path.isdir("/dataset") else "../"
        path = os.path.join(folder, csv_dataset_path)
        data = pd.read_csv(path, sep=",", index_col=0, parse_dates=True, decimal=".")
        print(f"Dataset {csv_dataset_path} loaded.")
        if resample_freq is not None:
            data = data.resample(resample_freq, label="left", closed="right").sum()
        return data

    return _thunk


@gin.configurable(module="code.datasets")
class DataCollection:
    def __init__(
        self,
        data_loader: Callable[[], pd.DataFrame],
    ):
        """Abstraction for the whole dataset, before slicing, cutting, splitting, etc.
        Args:
          data_loarder: a callable that returns a pd.DataFrame with a dataset.
        """
        self._data = data_loader()
        if self._data.index.freq:
            self._freq = self._data.index.freq.name
        else:
            # For Weather dataset, the following returns None.
            # This breaks the pipeline
            self._freq = pd.infer_freq(self._data.index)

    def freq(self, n_timesteps=1) -> pd.Timedelta:
        if self._freq == "B":
            return BDay(n_timesteps)
        else:
            # To handle the a situation like _freq = '10T', when
            # pd.Timedelta(n_timesteps, self._freq) throws
            # ValueError: invalid unit abbreviation: 15T.
            try:
                return pd.Timedelta(n_timesteps, unit=self._freq)
            except ValueError as e:
                return n_timesteps * pd.Timedelta(self._freq)

    @property
    def first_timestep(self):
        return self._data.index[0]

    @property
    def last_timestep(self):
        return self._data.index[-1]

    @property
    def num_timesteps(self):
        return self._data.shape[0]

    @property
    def data(self):
        return self._data


def _str_to_date(s):
    if type(s) is str:
        return pd.Timestamp(s)
    return s


@gin.configurable(module="code.datasets")
class Dataset:
    def __init__(
        self,
        data_full: DataCollection,
        series_length: int = 256,  # TODO: make it gin.REQUIRED
        start_date: str = gin.REQUIRED,
        train_window: int = gin.REQUIRED,
        eval_window: int = gin.REQUIRED,
        train_series: Tuple[int, int] = None,
        eval_series: Tuple[int, int] = None,
    ):
        """ " Abstraction for a dataset split into train/val

        Train data corresponds to an interval [self.train_start, self.eval_start-1].
        Eval data corresponds to an interval [self.eval_start, self.eval_end].

        Args:
          data_full: underlying data that we want to split,
          eval_window: Can be positive or 0. If positive, eval windows
              follows after training windows; if 0, eval window preceeds
              train windows by exactly `series_length`.
          series_length: Length of time series.
          train_series: a pair of ints denoting a range of timeseries ids used for
              training (or None),
          eval_series: analogously.
        """
        data = data_full.data  # [time, series]
        # TODO: make it more elegant
        self._freq = data_full._freq

        # Train data range.
        train_delta = data_full.freq(n_timesteps=train_window - 1)
        self.train_start = _str_to_date(start_date)
        self.train_end = self.train_start + train_delta

        # Eval data range.
        eval_delta = -1
        self.eval_horizon = series_length
        if eval_window > 0:
            eval_delta += train_window + eval_window
            self.eval_horizon = eval_window
        eval_delta = data_full.freq(n_timesteps=eval_delta)
        self.eval_end = self.train_start + eval_delta
        self.eval_start = self.eval_end + data_full.freq(n_timesteps=-series_length + 1)

        assert (
            self.train_start <= self.train_end and self.eval_start <= self.eval_end
        ), "Dates should be in chronological order."

        assert (
            min(self.eval_start, self.train_start) >= data_full.first_timestep
        ), f"Train start date {min(self.eval_start, self.train_start)} is before the \
        first timestep {data_full.first_timestep} in the dataset."

        assert max(
            self.eval_end, self.train_end
        ) <= data_full.last_timestep + data_full.freq(
            n_timesteps=1
        ), "Eval end date is after the last timestep in the dataset."

        # format trainset
        # INFO: data is a pd.DataFrame and [start_date:end_date] includes end_date.
        self.train_data = data[self.train_start : self.train_end].to_numpy().T
        if train_series is not None:
            (start, end) = train_series
            self.train_data = self.train_data[start:end]

        # [series, time]
        self.eval_data = data[self.eval_start : self.eval_end].to_numpy().T
        if eval_series is not None:
            (start, end) = eval_series  # range of series ids
            self.eval_data = self.eval_data[start:end]

        # format covariates, shape: [eval_window + train_window, num covariates (now = 3)]
        self.cov = self.__dynamic_covariates()

    def __dynamic_covariates(self):
        train_cov = generate_covariates(
            self.train_data.shape[1], self.train_start, self._freq
        )  # [train_window, 3]
        eval_cov = generate_covariates(
            self.eval_data.shape[1], self.eval_start, self._freq
        )  # [eval_window, 3]
        return np.concatenate([train_cov, eval_cov], axis=0)

    def covariates(self, series_idx: int, start: int, end: int) -> np.ndarray:
        # add series id to the last position of covariates,
        # (before, after) for each axis.
        return np.pad(self.cov[start:end], ((0, 0), (0, 1)), constant_values=series_idx)


def generate_covariates(ts_len, start, freq):
    """Returns covariates for a period starting at `start` and lasting `ts_len`."""
    # distance from the start in log-space, discretized
    inp = [np.floor(np.log2(np.arange(1, ts_len + 1)))]
    # cyclic covariates
    ts = pd.date_range(start, periods=ts_len, freq=freq)
    if freq == "H" or freq == "1H" or freq == "30min":
        # TODO: test if it would work better to have 48 categories for 30T.
        inp += [[t.dayofweek for t in ts], [t.hour for t in ts]]
    elif freq == "10T" or freq == "15T":
        minute_precision = int(freq[:-1])
        inp += [
            [t.dayofweek for t in ts],
            [t.hour * t.minute // minute_precision for t in ts],
        ]
    elif freq == "D" or freq == "B":
        inp += [[t.dayofweek for t in ts], [t.dayofyear for t in ts]]
    elif freq == "W":
        inp += [[t.week for t in ts]]
    elif freq == "M":
        inp += [[t.month for t in ts]]
    else:
        raise NotImplementedError(f"We do not support frequency {freq} yet!")
    return np.array(inp).T
