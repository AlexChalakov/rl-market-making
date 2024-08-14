import sys
import os
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple, Union

import numpy as np

from configurations import INDICATOR_WINDOW
from utils.EMA import ExponentialMovingAverage, load_ema

class Indicator(ABC):

    def __init__(self, label: str,
                 window: Union[int, None] = INDICATOR_WINDOW[0],
                 alpha: Union[List[float], float, None] = None):
        """
        Indicator constructor.
        :param window: (int) rolling window used for indicators
        :param alpha: (float) decay rate for EMA; if NONE, raw values returned
        """
        self._label = f"{label}_{window}"
        self.window = window
        if self.window is not None:
            self.all_history_queue = deque(maxlen=self.window + 1)
        else:
            self.all_history_queue = deque(maxlen=2)
        self.ema = load_ema(alpha=alpha)
        self._value = 0.

    def __str__(self):
        return f'Indicator.base() [ window={self.window}, ' \
               f'all_history_queue={self.all_history_queue}, ema={self.ema} ]'

    @abstractmethod
    def reset(self) -> None:
        """
        Clear values in indicator cache.
        :return: (void)
        """
        self._value = 0.
        self.all_history_queue.clear()

    @abstractmethod
    def step(self, **kwargs) -> None:
        """
        Update indicator with steps from the environment.
        :param kwargs: data values passed to indicators
        :return: (void)
        """
        if self.ema is None:
            pass
        elif isinstance(self.ema, ExponentialMovingAverage):
            self.ema.step(**kwargs)
        elif isinstance(self.ema, list):
            for ema in self.ema:
                ema.step(**kwargs)
        else:
            pass

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        """
        Calculate indicator value.
        :return: (float) value of indicator
        """
        pass

    @property
    def value(self) -> Union[List[float], float]:
        """
        Get indicator value for the current time step.
        :return: (scalar float)
        """
        if self.ema is None:
            return self._value
        elif isinstance(self.ema, ExponentialMovingAverage):
            return self.ema.value
        elif isinstance(self.ema, list):
            return [ema.value for ema in self.ema]
        else:
            return 0.

    @property
    def label(self) -> Union[List[str], str]:
        """
        Get indicator value for the current time step.
        :return: (scalar float)
        """
        if self.ema is None:
            return self._label
        elif isinstance(self.ema, ExponentialMovingAverage):
            return f"{self._label}_{self.ema.alpha}"
        elif isinstance(self.ema, list):
            return [f"{self._label}_{ema.alpha}" for ema in self.ema]
        else:
            raise ValueError(f"Error: EMA provided not valid --> {self.ema}")

    @property
    def raw_value(self) -> float:
        """
        Guaranteed raw value, if EMA is enabled.
        :return: (float) raw indicator value
        """
        return self._value

    @staticmethod
    def safe_divide(nom: float, denom: float) -> float:
        """
        Safely perform divisions without throwing a 'divide by zero' exception.
        :param nom: nominator
        :param denom: denominator
        :return: value
        """
        if denom == 0.:
            return 0.
        elif nom == 0.:
            return 0.
        else:
            return nom / denom

class IndicatorManager:
    __slots__ = ['indicators']

    def __init__(self):
        """
        Wrapper class to manage multiple indicators at the same time.
        """
        self.indicators = list()

    def add(self, name_and_indicator: Tuple[str, Union[Indicator, ExponentialMovingAverage]]) -> None:
        """
        Add indicator to the list to be managed.
        :param name_and_indicator: tuple(name, indicator)
        :return: (void)
        """
        self.indicators.append(name_and_indicator)

    def get_labels(self) -> list:
        """
        Get labels for each indicator being managed.
        :return: List of label names
        """
        labels = []
        for label, indicator in self.indicators:
            if isinstance(indicator, ExponentialMovingAverage):
                labels.append(f"{label}_EMA")
            else:
                indicator_label = indicator.label
                if isinstance(indicator_label, list):
                    labels.extend(indicator_label)
                else:
                    labels.append(indicator_label)
        return labels

    def step(self, **kwargs) -> None:
        """
        Update each indicator with new data.
        :param kwargs: Data passed to indicator for the update
        :return:
        """
        for name, indicator in self.indicators:
            if isinstance(indicator, ExponentialMovingAverage):
                indicator.step(value=kwargs.get('price'))
            elif isinstance(indicator, RSI):
                indicator.step(price=kwargs.get('price'))
            elif isinstance(indicator, TnS):
                indicator.step(buys=kwargs.get('buys'), sells=kwargs.get('sells'))

    def reset(self) -> None:
        """
        Reset all indicators being managed.
        :return: (void)
        """
        for (name, indicator) in self.indicators:
            indicator.reset()

    def get_value(self) -> List[float]:
        """
        Get all indicator values in the manager's inventory.
        :return: (list of floats) Indicator values for current time step
        """
        values = []
        for name, indicator in self.indicators:
            indicator_value = indicator.value
            if isinstance(indicator_value, list):
                values.extend(indicator_value)
            else:
                values.append(indicator_value)
        return values

class RSI(Indicator):
    """
    Price change momentum indicator. Note: Scaled to [-1, 1] and not [0, 100].
    """

    def __init__(self, **kwargs):
        super().__init__(label='rsi', **kwargs)
        self.last_price = None
        self.ups = self.downs = 0.

    def __str__(self):
        return f"RSI: [ last_price = {self.last_price} | " \
               f"ups = {self.ups} | downs = {self.downs} ]"

    def reset(self) -> None:
        """
        Reset the indicator.
        :return:
        """
        self.last_price = None
        self.ups = self.downs = 0.
        super().reset()

    def step(self, price: float) -> None:
        """
        Update indicator value incrementally.
        :param price: midpoint price
        :return:
        """
        if self.last_price is None:
            self.last_price = price
            return

        if np.isnan(price):
            print(f'Error: RSI.step() -> price is {price}')
            return

        if price == 0.:
            price_pct_change = 0.
        elif self.last_price == 0.:
            price_pct_change = 0.
        else:
            price_pct_change = round((price / self.last_price) - 1., 6)

        if np.isinf(price_pct_change):
            price_pct_change = 0.

        self.last_price = price

        if price_pct_change > 0.:
            self.ups += price_pct_change
        else:
            self.downs += price_pct_change

        self.all_history_queue.append(price_pct_change)

        # only pop off items if queue is done warming up
        if len(self.all_history_queue) <= self.window:
            return

        price_to_remove = self.all_history_queue.popleft()

        if price_to_remove > 0.:
            self.ups -= price_to_remove
        else:
            self.downs -= price_to_remove

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super().step(value=self._value)

    def calculate(self) -> float:
        """
        Calculate price momentum imbalance.
        :return: imbalance in range of [-1, 1]
        """
        mean_downs = abs(self.safe_divide(nom=self.downs, denom=self.window))
        mean_ups = self.safe_divide(nom=self.ups, denom=self.window)
        gain = mean_ups - mean_downs
        loss = mean_ups + mean_downs
        return self.safe_divide(nom=gain, denom=loss)
    
class TnS(Indicator):
    """
    Time and sales [trade flow] imbalance indicator
    """

    def __init__(self, **kwargs):
        super().__init__(label='tns', **kwargs)
        self.ups = self.downs = 0.

    def __str__(self):
        return f"TNS: ups={self.ups} | downs={self.downs}"

    def reset(self) -> None:
        """
        Reset indicator.
        """
        self.ups = self.downs = 0.
        super().reset()

    def step(self, buys: float, sells: float) -> None:
        """
        Update indicator with new transaction data.
        :param buys: buy transactions
        :param sells: sell transactions
        """
        self.ups += abs(buys)
        self.downs += abs(sells)
        self.all_history_queue.append((buys, sells))

        # only pop off items if queue is done warming up
        if len(self.all_history_queue) <= self.window:
            return

        buys_, sells_ = self.all_history_queue.popleft()
        self.ups -= abs(buys_)
        self.downs -= abs(sells_)

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super().step(value=self._value)

    def calculate(self) -> float:
        """
        Calculate trade flow imbalance.
        :return: imbalance in range of [-1, 1]
        """
        gain = round(self.ups - self.downs, 6)
        loss = round(self.ups + self.downs, 6)
        return self.safe_divide(nom=gain, denom=loss)