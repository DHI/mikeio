from abc import ABC, abstractmethod

import pandas as pd


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


class EquidistantTimeSeries(TimeSeries):
    @property
    @abstractmethod
    def timestep(self):
        pass
