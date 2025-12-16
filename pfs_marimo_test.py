"""Test PFS HTML representation in Marimo.

Run with: marimo edit pfs_marimo_test.py
"""

import marimo

__generated_with = "0.9.34"
app = marimo.App()


@app.cell
def __():
    import mikeio
    return (mikeio,)


@app.cell
def __(mikeio):
    pfs_text = """
    [FemEngineSW]
        version = 2.5
        [DOMAIN]
            mesh_file = |.\\mesh.mesh|
            coordinate_type = 'UTM-33'
            minimum_depth = 0.01
            number_of_layers = 10
        EndSect
        [TIME]
            start_time = 2020, 1, 1, 0, 0, 0
            time_step = 60.0
        EndSect
    EndSect
    """

    pfs = mikeio.PfsDocument.from_text(pfs_text)
    return pfs, pfs_text


@app.cell
def __(pfs):
    # This should display with rich HTML representation
    # Features:
    # - Search box
    # - Collapse/Expand All button
    # - Copy path buttons (📋)
    # - Color-coded values
    # - File path highlighting
    pfs.FemEngineSW
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo

    mo.md("""
    ## PFS HTML Representation in Marimo

    All features should work:
    - ✅ Search/filter box
    - ✅ Collapse/Expand All button
    - ✅ Path copy buttons (hover to see 📋)
    - ✅ Color-coded values
    - ✅ Dark mode (if Marimo is in dark mode)

    **Note:** Copy button requires clipboard permissions in your browser.
    """)
    return (mo,)


if __name__ == "__main__":
    app.run()
