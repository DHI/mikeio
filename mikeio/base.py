from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd

from .dataset import ItemInfo


class TimeSeries(ABC):
    @property
    @abstractmethod
    def start_time(self) -> pd.Timestamp:
        pass

    @property
    @abstractmethod
    def end_time(self) -> pd.Timestamp:
        pass

    @property
    @abstractmethod
    def n_timesteps(self) -> int:
        pass

    @property
    @abstractmethod
    def deletevalue(self) -> float:
        pass

    @property
    @abstractmethod
    def n_items(self) -> int:
        pass

    @property
    @abstractmethod
    def iteminfos(self) -> Sequence[ItemInfo]:
        pass


class EquidistantTimeSeries(TimeSeries):
    @property
    @abstractmethod
    def timestep(self):
        pass
