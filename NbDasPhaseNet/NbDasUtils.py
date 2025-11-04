# helper functions for reading and processing DAS data

# type: ignore
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.signal as sgn


def read_h5(
    h5_file,
) -> tuple[np.ndarray | h5.Dataset, float, float, np.ndarray, np.ndarray]:
    dataset = h5_file["Acoustic"]

    das_data = dataset
    attr = dataset.attrs

    print("DAS data shape:", das_data.shape)

    nTime = das_data.shape[0]
    nDist = das_data.shape[1]

    # Get the sampling interval and frequency
    spatial_sampling_interval = attr["SpatialSamplingInterval"]  # [m]
    time_sampling_interval = attr["TimeSamplingInterval(seconds)"]  # [s]

    time_series = np.arange(nTime) * time_sampling_interval
    dist_series = np.arange(nDist) * spatial_sampling_interval

    return (
        das_data,
        spatial_sampling_interval,
        time_sampling_interval,
        time_series,
        dist_series,
    )


def crop_das_distance_range(das_data, dist_series, dist_range):
    istart = np.argmax(dist_series >= dist_range[0]) - 1
    iend = np.argmax(dist_series > dist_range[1])
    crop_data = das_data[:, istart:iend]
    print("iStart:", istart, "iEnd:", iend)
    return crop_data, dist_series[istart:iend]


def get_trace_at_distance(das_data, distance, dist_series):
    idist = np.argmax(dist_series >= distance)
    trace = das_data[:, idist]
    return trace


def get_trace_at_time(das_data, time, time_series):
    itime = np.argmax(time_series >= time)
    trace = das_data[itime, :]
    return trace


def get_mean_trace(das_data, dist_series, dist_range=None):
    if dist_range == None:
        dist_range = (0, das_data.shape[1])

    crop_das_dist = crop_das_distance_range(das_data, dist_series, dist_range)
    trace = np.mean(crop_das_dist, axis=1)

    return trace


def normalize(das_data):
    return (das_data - np.mean(das_data)) / np.std(das_data)


def get_h5_files(input_dir, ignore_prefix=None):
    """
    Get a list of all h5 files in the input folder
    ---
    Parameters
    ----------
    input_dir : str
        Path to the input folder
    ignore_prefix : list[str]
        List of prefixes to ignore in the file names
    """
    import os

    # get a list of all files in the input folder
    all_files = os.listdir(input_dir)

    # filter out only the h5 files
    h5_files = [f for f in all_files if f.endswith(".h5")]

    # add the full path to the file names
    h5_files = [os.path.join(input_dir, f) for f in h5_files]

    # remove files with the specified prefixes in the file name
    if ignore_prefix is not None:
        h5_files = [
            f
            for f in h5_files
            if not any(
                os.path.basename(f).lower().startswith(prefix)
                for prefix in ignore_prefix
            )
        ]

    return h5_files


def crop_das_time_range(das_data, time_series, time_range):
    i_time_start = np.argmax(time_series >= time_range[0])
    i_time_end = np.argmax(time_series > time_range[1])
    return das_data[i_time_start:i_time_end, :], time_series[i_time_start:i_time_end]


# Bandpass filter
def sos_bandpass_filter(dat, norder, cutf1, cutf2, fs):
    sos = sgn.butter(norder, [cutf1, cutf2], "bandpass", fs=fs, output="sos")
    if len(dat.shape) == 1:
        rst = sgn.sosfilt(sos, dat)
    else:
        # loop over the columns of the data
        rst = np.zeros(dat.shape)
        # for i in range(dat.shape[1]):
        #     rst[:, i] = sgn.sosfilt(sos, dat[:, i])
        rst = sgn.sosfilt(sos, dat, axis=0)

    return rst

# lowpass filter
def sos_lowpass_filter(dat, norder, cutf, fs):
    sos = sgn.butter(norder, cutf, "lowpass", fs=fs, output="sos")
    if len(dat.shape) == 1:
        rst = sgn.sosfilt(sos, dat)
    else:
        # loop over the columns of the data
        rst = np.zeros(dat.shape)
        # for i in range(dat.shape[1]):
        #     rst[:, i] = sgn.sosfilt(sos, dat[:, i])
        rst = sgn.sosfilt(sos, dat, axis=0)

    return rst

# highpass filter
def sos_highpass_filter(dat, norder, cutf, fs):
    sos = sgn.butter(norder, cutf, "highpass", fs=fs, output="sos")
    if len(dat.shape) == 1:
        rst = sgn.sosfilt(sos, dat)
    else:
        # loop over the columns of the data
        rst = np.zeros(dat.shape)
        # for i in range(dat.shape[1]):
        #     rst[:, i] = sgn.sosfilt(sos, dat[:, i])
        rst = sgn.sosfilt(sos, dat, axis=0)

    return rst

# ===============================================================================
# region: visualization functions
# ===============================================================================


def plot_das_data(
    das_data,
    time_series,
    dist_series,
    time_interval,
    spatial_interval,
    vmax=10,
    vmin=-10,
    cmap="jet",
    title="",
    figsize=(8, 5),
):
    nTime = das_data.shape[0]
    nDist = das_data.shape[1]

    npts_time = 3000
    npts_dist = 3000

    dsf_time = np.max(
        [1, int(das_data.shape[0] / npts_time)]
    )  # downsample factor for time-axis
    dsf_dist = np.max(
        [1, int(das_data.shape[1] / npts_dist)]
    )  # downsample factor for distance-axis

    print("Time downsample factor: ", dsf_time)
    print("Distance downsample factor: ", dsf_dist)

    def setup_axes(title=title):
        plt.xlabel("Distance [m]")
        plt.ylabel("Time [s]")
        plt.title(title)

    def on_axis_limits_change(event):
        # replace the current plot with the resampled version
        # print("Axis limits changed")
        # print("New limits: ", plt.xlim(), ",", plt.ylim())

        i_time_start = int(np.min(plt.ylim()) / time_interval)
        i_time_end = int(np.max(plt.ylim()) / time_interval) + 1
        i_dist_start = int(np.min(plt.xlim()) / spatial_interval)
        i_dist_end = int(np.max(plt.xlim()) / spatial_interval) + 1

        # bound the indices to the data range
        i_time_start, i_time_end = np.clip([i_time_start, i_time_end], 0, nTime - 1)
        i_dist_start, i_dist_end = np.clip([i_dist_start, i_dist_end], 0, nDist - 1)

        # print("New indices: ", i_time_start, i_time_end, i_dist_start, i_dist_end)

        # recompute downsampling factors
        dsf_time = np.max([1, int((i_time_end - i_time_start) / npts_time)])
        dsf_dist = np.max([1, int((i_dist_end - i_dist_start) / npts_dist)])
        # print("New downsample factors: ", dsf_time, dsf_dist)

        resampled_das_data = das_data[
            i_time_start : i_time_end + dsf_time : dsf_time,
            i_dist_start : i_dist_end + dsf_dist : dsf_dist,
        ]

        cmap = plt.gci().cmap
        vmin, vmax = plt.gci().get_clim()

        cimg = plt.gca().images[0]

        ccolorbar = plt.gca().images[0].colorbar
        if ccolorbar != None:
            ccolorbar.remove()

        # remove the current image if it exists
        if cimg != None:
            cimg.remove()

        plt.imshow(
            resampled_das_data,
            aspect="auto",
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            extent=[
                dist_series[i_dist_start],
                dist_series[i_dist_end],
                time_series[i_time_end],
                time_series[i_time_start],
            ],
        )        
        setup_axes(title)
        # update the colorbar
        plt.gca().figure.colorbar(
            plt.gca().images[0], label=r"Strain rate ($\mu\epsilon/s$)"
        )
        plt.draw()

    # first, show the full data
    # plt.figure(figsize=figsize)
    plt.imshow(
        das_data[::dsf_time, ::dsf_dist],
        aspect="auto",
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        extent=[
            dist_series[0],
            dist_series[-1],
            time_series[-1],
            time_series[0],
        ],
    )
    setup_axes(title)
    plt.tight_layout()
    plt.draw()
    plt.colorbar(label=r"Strain rate ($\mu\epsilon/s$)")

    # hook event handlers for axis range changes
    plt.connect("button_release_event", on_axis_limits_change)
    plt.connect("key_release_event", on_axis_limits_change)


def plot_at_distance(das_data, distance, dist_series, time_series, basename=""):
    # plot a cross-section of the data
    trace = get_trace_at_distance(das_data, distance, dist_series)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time_series, y=trace, mode="lines", name="DAS data"))

    fig.update_layout(
        title=basename + " DAS data at distance " + str(distance) + " m",
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
    )
    fig.update_layout(margin=dict(l=10, r=10, b=5, t=30))
    fig.show()


def plot_fft_at_distance(das_data, distance, dist_series, time_series, basename=""):
    # plot a cross-section of the data
    trace = get_trace_at_distance(das_data, distance, dist_series)

    # calculate the FFT
    n = len(trace)

    yf = np.fft.fft(trace)
    yf = np.fft.fftshift(yf)
    yf = np.abs(yf) / n

    # calculate the frequency
    xf = np.fft.fftfreq(n, time_series[1] - time_series[0])
    xf = np.fft.fftshift(xf)

    fig = go.Figure()

    # plot the positive frequencies
    fig.add_trace(
        go.Scatter(x=xf[int(n / 2) :], y=yf[int(n / 2) :], mode="lines", name="FFT")
    )

    fig.update_layout(
        title=basename + " FFT of DAS data at distance " + str(distance) + " m",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude",
    )

    fig.update_layout(margin=dict(l=10, r=10, b=5, t=30))
    fig.show()


def show_random_locations(das_data, ilocations, i_delta_time, i_delta_dist, repeat=1):
    import matplotlib.pyplot as plt

    # randomly select 9 points from the signal locations
    # and show the cropped DAS data
    for _ in range(repeat):
        sel = np.random.choice(ilocations.shape[0], 9, replace=False)
        fig, ax = plt.subplots(3, 3, figsize=(5, 5))
        for i in range(3):
            for j in range(3):
                i_time = ilocations[sel[i * 3 + j], 0]
                i_dist = ilocations[sel[i * 3 + j], 1]

                ax[i, j].imshow(
                    normalize(
                        das_data[
                            i_time : i_time + i_delta_time,
                            i_dist : i_dist + i_delta_dist,
                        ].T
                    ),
                    cmap="gray",
                    aspect="auto",
                    vmin=-3,
                    vmax=3,
                )
                ax[i, j].axis("off")
        plt.show()
