import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfs2 - Snow Compare

        ## Using This Notebook

        -   Provides a simple differencing tool to compare two Dfs2 files, in
            this case MIKE SHE and MODIS snow cover
        -   Due to the ipywidgets controls which dynamically query Dfs2 files,
            this notebook requires local execution
        -   Modify the inputs section cell to provide file paths and then
            execute all cells
        -   Use the horizontal slider controls in the Analysis section to move
            back and forth in time for each Dfs2. A simple grid calculation will
            be performed showing which cells have lower, similar, or higher
            values, allowing you to identify areas of concern for further
            investigation

        ## Environment Setup

        -   The following packages were used at the time of development, and you
            may be able to use more recent versions:
            -   ipykernel 6.29.0
            -   mikeio 1.6.3
            -   matplotlib 3.8.2
            -   ipywidgets 8.1.5
        """
    )
    return


@app.cell
def _():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import mikeio
    import ipywidgets as widgets
    return mikeio, mpl, np, plt, widgets


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Inputs Section
        """
    )
    return


@app.cell
def _(mikeio):
    mikeSheSnowDfs2Path = "../tests/testdata/MikeSheExtract.dfs2"
    modisSnowDfs2Path = "../tests/testdata/ModisExtract.dfs2"

    mikeSheSnowDfs2 = mikeio.read(mikeSheSnowDfs2Path)
    modisSnowDfs2 = mikeio.read(modisSnowDfs2Path)
    return (
        mikeSheSnowDfs2,
        mikeSheSnowDfs2Path,
        modisSnowDfs2,
        modisSnowDfs2Path,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Functions Section
        """
    )
    return


@app.cell
def _(mikeSheSnowDfs2, modisSnowDfs2, mpl, np, plt, widgets):
    def sliders_changed(msheSlider, modisSlider):
        modisItemName = 'Snow Cover'
        msheItemName = 'Fraction of cell area covered by Snow'

        timeIndex = modisSnowDfs2[modisItemName].time.get_loc(modisSlider)
        modisSingleTimestep = modisSnowDfs2[modisItemName].isel(
            time=timeIndex) / 100.0

        timeIndex = mikeSheSnowDfs2[msheItemName].time.get_loc(msheSlider)
        mikeSheSingleTimestep = mikeSheSnowDfs2[msheItemName].isel(time=timeIndex)

        # use nans in MSHE to set nans in MODIS to mask out catchment
        modisSingleTimestep.values = np.where(np.isnan(
            mikeSheSingleTimestep.values), mikeSheSingleTimestep.values, modisSingleTimestep.values)
        modisSingleDataArray = modisSingleTimestep.to_xarray()
        modisSingleDataArray.attrs['units'] = 'Fraction'

        # 3 column layout
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
        modisSingleDataArray.plot.pcolormesh(
            ax=ax1, vmin=0.0, vmax=1.0, cmap="jet")
        ax1.set_title('MODIS')

        mikeSheSingleDataArray = mikeSheSingleTimestep.to_xarray()
        mikeSheSingleDataArray.attrs['units'] = 'Fraction'
        mikeSheSingleDataArray.plot.pcolormesh(
            ax=ax2, vmin=0.0, vmax=1.0, cmap="jet")
        ax2.set_title('MIKE SHE')

        diff = modisSingleTimestep.copy()
        diff.values = modisSingleTimestep.values - mikeSheSingleTimestep.values

        categories = ['MSHE Higher', 'Similar', 'Similar', 'MODIS Higher']
        colors = ['yellow', 'green', 'red']
        boundaries = [-1.0, -0.1, 0.1, 1.0]

        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)

        diffPlot = diff.to_xarray().plot.pcolormesh(ax=ax3, cmap=cmap, norm=norm)
        cbar = diffPlot.colorbar
        cbar.set_ticklabels(categories)
        cbar.set_label(None)

        ax3.set_title('Difference (MODIS - MIKE SHE)')
        plt.tight_layout()


    style = {'description_width': 'initial'}

    msheSlider = widgets.SelectionSlider(
        options=mikeSheSnowDfs2.time,
        value=mikeSheSnowDfs2.time[0],
        description='MSHE Timestep',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style,
        layout=widgets.Layout(width='800px', margin='0 30px 0 0 ')
    )

    # constrain the modis timesteps to the mshe, as modis covers far more
    modis_time_slicer = modisSnowDfs2.time.slice_indexer(
        start=mikeSheSnowDfs2.time[0], end=mikeSheSnowDfs2.time[-1])
    filtered_modis_times = modisSnowDfs2.time[modis_time_slicer]

    modisSlider = widgets.SelectionSlider(
        options=filtered_modis_times,
        value=filtered_modis_times[0],
        description='MODIS Timestep',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style,
        layout=widgets.Layout(width='800px', margin='0 30px 0 0 ')
    )
    return (
        filtered_modis_times,
        modisSlider,
        modis_time_slicer,
        msheSlider,
        sliders_changed,
        style,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Analysis Section
        """
    )
    return


@app.cell
def _(display, modisSlider, msheSlider, sliders_changed, widgets):
    ui = widgets.HBox([modisSlider, msheSlider])
    out = widgets.interactive_output(sliders_changed, {'msheSlider' : msheSlider, 'modisSlider' : modisSlider})
    display(ui, out)
    return out, ui


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

