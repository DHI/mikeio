import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # PFS 

        The support for PFS files have been extended with MIKE IO release 1.2. It was previously only possible to *read* PFS files. It is now also possible to *modify* and *create* new PFS files. 
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Read
        """
    )
    return


@app.cell
def _(mikeio):
    pfs = mikeio.read_pfs("../tests/testdata/pfs/lake.sw")
    pfs
    return (pfs,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The "target" (root section) is in this case called FemEngineSW. `pfs.FemEngineSW` is a PfsSection object that contains other PfsSection objects. Let's print the names of it's subsections:
        """
    )
    return


@app.cell
def _(pfs):
    pfs.FemEngineSW.keys()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        It is possible to navigate to each section and keyword in the pfs file:
        """
    )
    return


@app.cell
def _(pfs):
    pfs.FemEngineSW.DOMAIN.file_name
    return


@app.cell
def _(pfs):
    pfs.FemEngineSW.MODULE_SELECTION
    return


@app.cell
def _(pfs):
    pfs.FemEngineSW.MODULE_SELECTION.mode_of_spectral_wave_module
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        If you are unsure the name of a section, it is also possible to search for a specific string in the file, to find the name of a specific section.

        In the example below we do an case-insensitive search for the string 'charnock', which occurs at 6 different places in this file.
        """
    )
    return


@app.cell
def _(pfs):
    pfs.search("charnock")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The same search can be done at any level of the hierarchy, i.e. to search only within the OUTPUTS section:
        """
    )
    return


@app.cell
def _(pfs):
    pfs.FemEngineSW.SPECTRAL_WAVE_MODULE.OUTPUTS.search("charnock")
    return


@app.cell
def _(pfs):
    pfs.FemEngineSW.SPECTRAL_WAVE_MODULE.WIND.search("charnock")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        MIKE FM PFS files has a specific structure and active FM modules can be accessed by an alias on the Pfs object. In this case, `pfs.SW` can be used as a short-hand for `pfs.FemEngineSW.SPECTRAL_WAVE_MODULE`.
        """
    )
    return


@app.cell
def _(pfs):
    pfs.SW.SPECTRAL.number_of_directions
    return


@app.cell
def _(pfs):
    pfs.SW.SPECTRAL.maximum_threshold_frequency
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Enumerated sections (e.g. [OUTPUT_1], [OUTPUT_2], ...) can be outputted in tabular form (dataframe).
        """
    )
    return


@app.cell
def _(pfs):
    df = pfs.SW.OUTPUTS.to_dataframe()
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Modify

        The PfsSection object can be modified. Existing values can be changes, new key-value pairs can be added, subsections can added or removed. 
        """
    )
    return


@app.cell
def _(pfs):
    pfs.SW.SPECTRAL.number_of_directions = 32
    return


@app.cell
def _(pfs):
    pfs.SW.SPECTRAL
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Add a new keyword
        """
    )
    return


@app.cell
def _(pfs):
    pfs.SW.SPECTRAL["new_keyword"] = "new_value"
    return


@app.cell
def _(pfs):
    pfs.SW.SPECTRAL
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Add a section

        Let's create an additional output, by copying OUTPUT_4 and modifying some parameters.
        """
    )
    return


@app.cell
def _(pfs):
    pfs.SW.OUTPUTS.number_of_outputs = pfs.SW.OUTPUTS.number_of_outputs + 1
    return


@app.cell
def _(pfs):
    new_output = pfs.SW.OUTPUTS.OUTPUT_4.copy()
    return (new_output,)


@app.cell
def _(new_output):
    new_output.file_name = 'spectrum_x10km_y40km.dfsu'
    new_output.POINT_1.x = 10000
    new_output.POINT_1.y = 40000
    return


@app.cell
def _(new_output, pfs):
    pfs.SW.OUTPUTS["OUTPUT_5"] = new_output
    return


@app.cell
def _(pfs):
    pfs.SW.OUTPUTS.keys()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Output

        The Pfs object can be written to pfs file, but can also be exported to a dictionary (which in turn can be written to a yaml or json file).
        """
    )
    return


@app.cell
def _(pfs):
    pfs.write("lake_modified.pfs")
    return


@app.cell
def _(pfs):
    pfs.to_dict()
    return


@app.cell
def _(pfs):
    # write to yaml file
    import yaml
    yaml.dump(pfs.to_dict(), open('lake_modified.yaml', 'w+'))
    return (yaml,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Create

        A PFS file can also be created from a dictionary, like this:
        """
    )
    return


@app.cell
def _():
    setup = {
        "Name": "Extract that",
        "InputFileName": "|random.dfs1|",
        "FirstTimeStep": 0,
        "LastTimeStep": 99,
        "X": 2,
        "OutputFileName": "|.\\out2.dfs0|",
    }
    t1_t0 = {"CLSID": "t1_t0.dll", "TypeName": "t1_t0", "Setup": setup}
    t1_t0
    return setup, t1_t0


@app.cell
def _(mikeio, t1_t0):
    pfs_1 = mikeio.PfsDocument({'t1_t0': t1_t0})
    pfs_1
    return (pfs_1,)


@app.cell
def _(pfs_1):
    pfs_1.write('extract_point.mzt')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clean up
        """
    )
    return


@app.cell
def _():
    import os
    os.remove("lake_modified.pfs")
    os.remove('lake_modified.yaml')
    os.remove("extract_point.mzt")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

