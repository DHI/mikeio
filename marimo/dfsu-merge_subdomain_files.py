import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Merging subdomain dfsu files

        during simulation MIKE will commonly split simulation files into subdomains and output results with a p_# suffix. This script will merge dfsu files of this type into a single file.

        Note: Below implementation considers a 2D dfsu file. For 3D dfsu file, the script needs to be modified accordingly.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""**import libraries**""")
    return


@app.cell
def _():
    import mikeio 
    import numpy as np
    from mikeio.spatial import GeometryFM2D
    return GeometryFM2D, mikeio, np


@app.cell
def _(mikeio):
    # (optional) check first file, items etc. 
    mikeio.open("../tests/testdata/SimA_HD_p0.dfsu")
    return


@app.cell
def _(mo):
    mo.md(r"""**choose items to process**""")
    return


@app.cell
def _():
    # choose items to process (when in doubt look at one of the files you want to process with mikeio.open)
    items = ["Surface elevation", "Current speed", "Current direction"]
    return (items,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **read files**

        Option A: automatically find all with _p# suffix
        """
    )
    return


@app.cell
def _(items, mikeio):
    import glob
    import os
    basename = '../tests/testdata/SimA_HD'

    def find_dfsu_files(basename):
        pattern = f'{basename}_p*.dfsu'
        files = sorted(glob.glob(pattern))
        if not files:
            raise ValueError(f'No files found matching the pattern: {pattern}')
        return files
    _dfs_files = find_dfsu_files(basename)
    print(f'Found {len(_dfs_files)} files:')
    for file in _dfs_files:
        print(f'  - {os.path.basename(file)}')
    dfs_list = [mikeio.read(file, items=items) for file in _dfs_files]
    return basename, dfs_list, file, find_dfsu_files, glob, os


@app.cell
def _(mo):
    mo.md(r"""Option B: manually select files""")
    return


@app.cell
def _(items, mikeio):
    _dfs_files = ['../tests/testdata/SimA_HD_p0.dfsu', '../tests/testdata/SimA_HD_p1.dfsu', '../tests/testdata/SimA_HD_p2.dfsu', '../tests/testdata/SimA_HD_p3.dfsu']
    dfs_list_1 = [mikeio.read(file, items=items) for file in _dfs_files]
    return (dfs_list_1,)


@app.cell
def _(mo):
    mo.md(r"""**extract data of all subdomains**""")
    return


@app.cell
def _(dfs_list_1, items, np):
    data_dict = {item: [] for item in items}
    time_steps = dfs_list_1[0][items[0]].time
    for item in items:
        for i in range(len(time_steps)):
            combined_data = np.concatenate([dfs[item].values[i, :] for dfs in dfs_list_1])
            data_dict[item].append(combined_data)
        data_dict[item] = np.array(data_dict[item])
    merged_data = np.array([data_dict[item] for item in items])
    return combined_data, data_dict, i, item, merged_data, time_steps


@app.cell
def _(mo):
    mo.md(r"""**merge geometry of all subdomains**""")
    return


@app.cell
def _(GeometryFM2D, dfs_list_1, np):
    geometries = [dfs.geometry for dfs in dfs_list_1]
    combined_node_coordinates = []
    combined_element_table = []
    node_offset = 0
    for geom in geometries:
        current_node_coordinates = geom.node_coordinates
        current_element_table = geom.element_table
        combined_node_coordinates.extend(current_node_coordinates)
        adjusted_element_table = [element + node_offset for element in current_element_table]
        combined_element_table.extend(adjusted_element_table)
        node_offset = node_offset + len(current_node_coordinates)
    combined_node_coordinates = np.array(combined_node_coordinates)
    combined_element_table = np.array(combined_element_table, dtype=object)
    projection = geometries[0]._projstr
    combined_geometry = GeometryFM2D(node_coordinates=combined_node_coordinates, element_table=combined_element_table, projection=projection)
    return (
        adjusted_element_table,
        combined_element_table,
        combined_geometry,
        combined_node_coordinates,
        current_element_table,
        current_node_coordinates,
        geom,
        geometries,
        node_offset,
        projection,
    )


@app.cell
def _(combined_geometry):
    combined_geometry.plot()
    return


@app.cell
def _(mo):
    mo.md(r"""**merge everything into dataset**""")
    return


@app.cell
def _(combined_geometry, items, merged_data, mikeio, time_steps):
    ds_out = mikeio.Dataset.from_numpy(
        data=merged_data,  # n_items, timesteps, n_elements
        items=items, 
        time=time_steps,
        geometry=combined_geometry
    )
    return (ds_out,)


@app.cell
def _(ds_out, items):
    ds_out[items[0]].sel(time=1).plot() # plot the first time step of the first item
    return


@app.cell
def _(mo):
    mo.md(r"""**write output to single file**""")
    return


@app.cell
def _(ds_out):
    output_file = "area_merged.dfsu"
    ds_out.to_dfs(output_file)
    return (output_file,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
