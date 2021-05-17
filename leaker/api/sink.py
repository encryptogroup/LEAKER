"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from abc import ABC, abstractmethod


class DataSink(ABC):
    """A data sink to write performance data of evaluated attacks to."""

    @abstractmethod
    def register_series(self, series_id: str) -> None:
        """
        Reqisters an attack. It must be called before using offer_data for that series for the first time.

        Parameters
        ----------
        series_id : str
            the name of the attack
        """
        raise NotImplementedError

    @abstractmethod
    def offer_data(self, series_id: str, user_id: int, known_data_rate: float, recovery_rate: float) -> None:
        """
        Passes a new data point the the data sink.

        Parameters
        ----------
        series_id : str
            the name of the attack
        user_id : int
            the user id
        known_data_rate : float
            the known data rate the attack was evaluated on
        recovery_rate : float
            the resulting recovery rate
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Executes all remaining write operations."""
        raise NotImplementedError
