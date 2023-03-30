import numpy as np


def plot_2dspectrum(
    spectrum,
    frequencies,
    directions,
    plot_type="contourf",
    title=None,
    label=None,
    cmap="Reds",
    vmin=1e-5,
    vmax=None,
    r_as_periods=True,
    rmin=None,
    rmax=None,
    levels=None,
    figsize=(7, 7),
    add_colorbar=True,
):
    """
    Plot spectrum in polar coordinates

    Parameters
    ----------
    spectrum: np.array

        spectral values as 2d array with dimensions: directions, frequencies
    plot_type: str, optional
        type of plot: 'contour', 'contourf', 'patch', 'shaded',
        by default: 'contourf'
    title: str, optional
        axes title
    label: str, optional
        colorbar label (or title if contour plot)
    cmap: matplotlib.cm.cmap, optional
        colormap, default Reds
    vmin: real, optional
        lower bound of values to be shown on plot, default: 1e-5
    vmax: real, optional
        upper bound of values to be shown on plot, default:None
    r_as_periods: bool, optional
        show radial axis as periods instead of frequency, default: True
    rmin: float, optional
        mininum frequency/period to be shown, default: None
    rmax: float, optional
        maximum frequency/period to be shown, default: None
    levels: int, list(float), optional
        for contour plots: how many levels, default:10
        or a list of discrete levels e.g. [0.03, 0.04, 0.05]
    figsize: (float, float), optional
        specify size of figure, default (7, 7)
    add_colorbar: bool, optional
        Add colorbar to plot, default True

    Returns
    -------
    <matplotlib.axes>

    Examples
    --------
    >>> dfs = Dfsu("area_spectrum.dfsu")
    >>> ds = dfs.read(items="Energy density")
    >>> spectrum = ds[0][0, 0, :, :] # first timestep, element 0
    >>> dfs.plot_spectrum(spectrum, plot_type="patch")

    >>> dfs.plot_spectrum(spectrum, rmax=9, title="Wave spectrum T<9s")
    """

    import matplotlib.pyplot as plt

    if (frequencies is None or len(frequencies) <= 1) and (
        directions is None or len(directions) <= 1
    ):
        raise ValueError("plot_2dspectrum() is only supported for full spectral data")

    dirs = np.radians(directions)
    freq = frequencies

    inverse_r = r_as_periods
    if inverse_r:
        freq = 1.0 / np.flip(freq)
        spectrum = np.fliplr(spectrum)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    ddir = dirs[1] - dirs[0]

    def is_circular(dir):
        dir_diff = np.mod(dir[0], 2 * np.pi) - np.mod(dir[-1] + ddir, 2 * np.pi)
        return np.abs(dir_diff) < 1e-6

    if is_circular(dirs):
        # append last directional slice at the end
        dirs = np.append(dirs, dirs[-1] + ddir)
        spectrum = np.concatenate((spectrum, spectrum[0:1, :]), axis=0)

    # up-sample directions
    if plot_type in ("shaded", "contour", "contourf"):
        # more smoother plotting
        factor = 4
        dir2 = np.linspace(dirs[0], dirs[-1], (len(dirs) - 1) * factor + 1)
        spec2 = np.zeros(shape=(len(dir2), len(freq)))
        for j in range(len(freq)):
            spec2[:, j] = np.interp(dir2, dirs, spectrum[:, j])
        dirs = dir2.copy()
        spectrum = spec2.copy()

    if vmin is None:
        vmin = np.nanmin(spectrum)
    if vmax is None:
        vmax = np.nanmax(spectrum)
    if levels is None:
        levels = 10
        n_levels = 10
    if np.isscalar(levels):
        n_levels = levels
        levels = np.linspace(vmin, vmax, n_levels)

    if plot_type != "shaded":
        spectrum[spectrum < vmin] = np.nan

    if plot_type == "contourf":
        colorax = ax.contourf(
            dirs, freq, spectrum.T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
        )
    elif plot_type == "contour":
        colorax = ax.contour(
            dirs, freq, spectrum.T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
        )
        # ax.clabel(colorax, fmt="%1.2f", inline=1, fontsize=9)
        if label is not None:
            ax.set_title(label)

    elif plot_type in ("patch", "shaded", "box"):
        shading = "gouraud" if plot_type == "shaded" else "auto"
        ax.grid(False)  # Remove major grid
        colorax = ax.pcolormesh(
            dirs,
            freq,
            spectrum.T,
            shading=shading,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.grid("on")
    else:
        raise ValueError(
            f"plot_type '{plot_type}' not supported (contour, contourf, patch, shaded)"
        )

    # TODO: optional
    ax.set_thetagrids(
        [0.0, 45, 90.0, 135, 180.0, 225, 270.0, 315],
        labels=["N", "N-E", "E", "S-E", "S", "S-W", "W", "N-W"],
    )

    # if r_axis_as_periods:
    #     Ts = [8, 6, 5, 4, 3.5, 3, 2.5, 2]
    #     ax.set_rgrids(
    #         1.0 / np.array(Ts),
    #         labels=["8s", "6s", "5s", "4s", "3.5s", "3s", "2.5s", "2s"],
    #     )
    #     # , fontsize=12, angle=180)

    # ax.grid(True, which='minor', axis='both', linestyle='-', color='0.8')
    # ax.set_xticks(dfs.directions, minor=True);

    if rmin is not None:
        ax.set_rmin(rmin)
    if rmax is not None:
        ax.set_rmax(rmax)

    if add_colorbar:
        cbar = fig.colorbar(colorax)
        if label is None:
            label = "Energy Density [m*m/Hz/deg]"
        cbar.set_label(label, rotation=270)
        cbar.ax.get_yaxis().labelpad = 30

    if title is not None:
        ax.set_title(title)

    return ax


def calc_m0_from_spectrum(spec, f, dir=None, tail=True):
    if f is None:
        nd = len(dir)
        dtheta = (dir[-1] - dir[0]) / (nd - 1)
        return np.sum(spec, axis=-1) * dtheta * np.pi / 180.0
    df = _f_to_df(f)

    if dir is None:
        ee = spec
    else:
        nd = len(dir)
        dtheta = (dir[-1] - dir[0]) / (nd - 1)
        ee = np.sum(spec, axis=-2) * dtheta

    m0 = np.dot(ee, df)
    if tail:
        m0 = m0 + ee[..., -1] * f[-1] * 0.25
    return m0


def _f_to_df(f):
    """Frequency bins for equidistant or logrithmic frequency axis"""
    if np.isclose(np.diff(f).min(), np.diff(f).max()):
        # equidistant frequency bins
        return (f[1] - f[0]) * np.ones_like(f)
    else:
        # logarithmic frequency bins
        freq_factor = f[1] / f[0]
        fm1 = np.insert(f, 0, f[0] / freq_factor)
        fp1 = np.append(f, f[-1] * freq_factor)
        return 0.5 * (np.diff(fm1) + np.diff(fp1))
