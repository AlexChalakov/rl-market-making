import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Union

import numpy as np
import pandas as pd

from configurations import LOGGER

# Calculate Exponential moving average in O(1) time.
class ExponentialMovingAverage(object):
    __slots__ = ['alpha', '_value']

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._value = None

    def __str__(self):
        return f'ExponentialMovingAverage: [ alpha={self.alpha} | value={self._value} ]'

    def step(self, value: float) -> None:
        """
        Update EMA at every time step.
        :param value: price at current time step
        :return: (void)
        """
        if self._value is None:
            self._value = value
            return

        self._value = (1. - self.alpha) * value + self.alpha * self._value

    @property
    def value(self) -> float:
        """
        EMA value of data.
        :return: (float) EMA smoothed value
        """
        return self._value

    def reset(self) -> None:
        """
        Reset EMA.
        :return: (void)
        """
        self._value = None


def load_ema(alpha: Union[List[float], float, None]) -> \
        Union[List[ExponentialMovingAverage], ExponentialMovingAverage, None]:
    """
    Set exponential moving average smoother.
    :param alpha: decay rate for EMA
    :return: (var) EMA
    """
    if alpha is None:
        # print("EMA smoothing DISABLED")
        return None
    elif isinstance(alpha, float):
        LOGGER.info(f"EMA smoothing ENABLED: {alpha}")
        return ExponentialMovingAverage(alpha=alpha)
    elif isinstance(alpha, list):
        LOGGER.info(f"EMA smoothing ENABLED: {alpha}")
        return [ExponentialMovingAverage(alpha=a) for a in alpha]
    else:
        raise ValueError(f"_load_ema() --> unknown alpha type: {type(alpha)}")


def apply_ema_all_data(
        ema: Union[List[ExponentialMovingAverage], ExponentialMovingAverage, None],
        data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply exponential moving average to entire data set in a single batch.
    :param ema: EMA handler; if None, no EMA is applied
    :param data: data set to smooth
    :return: (pd.DataFrame) smoothed data set, if ema is provided
    """
    if ema is None:
        return data

    smoothed_data = []
    labels = data.columns.tolist()

    if isinstance(ema, ExponentialMovingAverage):
        for value in data['Midpoint'].values:  # Apply EMA to 'Midpoint' column
            ema.step(value=value)
            smoothed_data.append(ema.value)
        data['Midpoint_EMA'] = smoothed_data  # Add new column with EMA values
        return data
    elif isinstance(ema, list):
        # Assuming you want to apply multiple EMA values to different columns
        for column in data.columns:
            smoothed_data = []
            for e in ema:
                for value in data[column].values:
                    e.step(value=value)
                    smoothed_data.append(e.value)
            data[f'{column}_EMA'] = smoothed_data
        return data
    else:
        raise ValueError(f"apply_ema_all_data() --> unknown ema type: {type(ema)}")



def reset_ema(ema: Union[List[ExponentialMovingAverage], ExponentialMovingAverage, None]) -> \
        Union[List[ExponentialMovingAverage], ExponentialMovingAverage, None]:
    """
    Reset the EMA smoother.
    :param ema:
    :return:
    """
    if ema is None:
        pass
    elif isinstance(ema, ExponentialMovingAverage):
        ema.reset()
        LOGGER.info("Reset EMA data.")
    elif isinstance(ema, list):
        for e in ema:
            e.reset()
        LOGGER.info("Reset EMA data.")
    return ema