from abc import ABC, abstractmethod


class TimeSeries(ABC):
    @property
    @abstractmethod
    def start_time(self):
        pass

    @property
    @abstractmethod
    def end_time(self):
        pass

    @property
    @abstractmethod
    def n_timesteps(self):
        pass

    @property
    @abstractmethod
    def deletevalue(self):
        pass

    @property
    @abstractmethod
    def n_items(self):
        pass

    @property
    @abstractmethod
    def items(self):
        pass


class EquidistantTimeSeries(TimeSeries):
    @property
    @abstractmethod
    def timestep(self):
        pass
