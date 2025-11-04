import numpy as np
import h5py
from typing import Optional
import NbDasUtils as utils

class DasData:
    """
    Class to store a single DAS measurement data and acquisition parameters.
    """

    def __init__(
        self,
        data: np.ndarray | h5py.Dataset,
        spatial_interval: float,
        time_interval: float,
        distance_array: np.ndarray,
        time_array: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        """
        Initialize the DasData object with data, spatial interval, time interval, and metadata.

        :param data: 2D numpy array of DAS measurement data (time x spatial).
        :param spatial_interval: Spatial interval in meters.
        :param time_interval: Time interval in seconds.
        :param time_array: 1D numpy array of time values.
        :param distance_array: 1D numpy array of distance values.
        :param metadata: Dictionary containing metadata information.
        """

        # Verify the input shapes
        if data is not None:
            if not isinstance(data, np.ndarray) and not isinstance(data, h5py.Dataset):
                raise ValueError("Data must be a numpy array or HDF5 dataset.")
            if data.ndim != 2:
                raise ValueError("Data must be a 2D array.")
            if time_array is not None and data.shape[0] != len(time_array):
                raise ValueError(
                    "Time array length must match the first dimension of data."
                )
            if distance_array is not None and data.shape[1] != len(distance_array):
                raise ValueError(
                    "Distance array length must match the second dimension of data."
                )

        self._data = data
        self._spatial_interval = spatial_interval
        self._time_interval = time_interval
        self._time_array = time_array
        self._distance_array = distance_array
        self._metadata = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        """String representation of the DasData object."""
        return (
            f"DasData(num_samples={self.num_samples}, "
            f"num_channels={self.num_channels}, "
            f"spatial_interval={self.spatial_interval:.6f}, "
            f"time_interval={self.time_interval:.6f})"
        )
    
    @property
    def data(self) -> np.ndarray:
        """Get the DAS measurement data."""
        return self._data

    @property
    def spatial_interval(self) -> float:
        """Get the spatial interval in meters."""
        return self._spatial_interval

    @property
    def time_interval(self) -> float:
        """Get the time interval in seconds."""
        return self._time_interval

    @property
    def time_array(self) -> np.ndarray:
        """Get the time array."""
        return self._time_array

    @property
    def distance_array(self) -> np.ndarray:
        """Get the distance array."""
        return self._distance_array

    @property
    def metadata(self) -> dict:
        """Get the metadata dictionary."""
        return self._metadata.copy()

    @property
    def interrogation_rate(self) -> float:
        """Get the interrogation rate in Hz."""
        return 1 / self.time_interval

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return self.data.shape[1]

    @property
    def num_timeSamples(self) -> int:
        """Get the number of samples in time axis."""
        return self.data.shape[0]

    def bandpass_filter(
        self,
        lowcut: float,
        highcut: float,
        order: int = 4,
    ):
        """
        Apply a bandpass filter to the data.
        Caution: This method modifies the data in place.

        :param data: Input data to be filtered.
        :param lowcut: Low cutoff frequency in Hz.
        :param highcut: High cutoff frequency in Hz.
        :param fs: Sampling frequency in Hz.
        :param order: Order of the filter.
        :return: Filtered data.
        """
        import scipy.signal as sgn

        sos = sgn.butter(
            order,
            [lowcut, highcut],
            "bandpass",
            fs=self.interrogation_rate,
            output="sos",
        )

        # modify the data in place
        self._data = sgn.sosfilt(sos, self._data, axis=0)

    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 4,
    ):
        """
        Apply a lowpass filter to the data.
        Caution: This method modifies the data in place.

        :param data: Input data to be filtered.
        :param cutoff: Cutoff frequency in Hz.
        :param fs: Sampling frequency in Hz.
        :param order: Order of the filter.
        :return: Filtered data.
        """
        import scipy.signal as sgn

        sos = sgn.butter(
            order,
            cutoff,
            "lowpass",
            fs=self.interrogation_rate,
            output="sos",
        )

        # modify the data in place
        self._data = sgn.sosfilt(sos, self._data, axis=0)

    
    def highpass_filter(
        self,
        cutoff: float,
        order: int = 4,
    ):
        """
        Apply a highpass filter to the data.
        Caution: This method modifies the data in place.

        :param data: Input data to be filtered.
        :param cutoff: Cutoff frequency in Hz.
        :param fs: Sampling frequency in Hz.
        :param order: Order of the filter.
        :return: Filtered data.
        """
        import scipy.signal as sgn

        sos = sgn.butter(
            order,
            cutoff,
            "highpass",
            fs=self.interrogation_rate,
            output="sos",
        )

        # modify the data in place
        self._data = sgn.sosfilt(sos, self._data, axis=0)


    def plot_data(self, vmin=-1.0, vmax=1.0, cmap="jet", title=""):
        """Plot the DAS data."""
        utils.plot_das_data(self.data, self.time_array, self.distance_array, 
                            self.time_interval, self.spatial_interval, 
                            vmin=vmin, vmax=vmax, cmap=cmap, title=title)