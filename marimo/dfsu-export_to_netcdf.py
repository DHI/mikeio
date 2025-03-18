import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Export to netcdf
        * Read data from dfsu file
        * Convert to xarray dataset
        * Write to netcdf file
        """
    )
    return


@app.cell
def _():
    # pip install mikeio xarray netcdf4
    return


@app.cell
def _():
    import os
    import mikeio
    import xarray as xr
    return mikeio, os, xr


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/oresund_sigma_z.dfsu")
    ds
    return (ds,)


@app.cell
def _(ds):
    nc = ds.geometry.node_coordinates
    xn = nc[:,0]
    yn = nc[:,1]
    zn = nc[:,2]

    ec = ds.geometry.element_coordinates
    xe = ec[:,0]
    ye = ec[:,1]
    ze = ec[:,2]
    return ec, nc, xe, xn, ye, yn, ze, zn


@app.cell
def _(ds, ec, nc, xe, xn, xr, ye, ze, zn):
    # Time
    time = ds.time

    # Node based data
    node_ids = list(range(len(nc)))
    z_dynamic = ds._zn
    xn_da = xr.DataArray(xn, coords=[node_ids], dims=["nodes"], attrs={'units': 'meter'})
    yn_da = xr.DataArray(xn, coords=[node_ids], dims=["nodes"], attrs={'units': 'meter'})
    zn_da = xr.DataArray(zn, coords=[node_ids], dims=["nodes"], attrs={'units': 'meter'})
    z_dyn_da = xr.DataArray(z_dynamic, coords =[time,node_ids],dims=["time", "nodes"], attrs={'units': 'meter'})

    # Element based data
    el_ids = list(range(len(ec)))
    xe_da = xr.DataArray(xe, coords=[el_ids], dims=["elements"], attrs={'units': 'meter'})
    ye_da = xr.DataArray(ye, coords=[el_ids], dims=["elements"], attrs={'units': 'meter'})
    ze_da = xr.DataArray(ze, coords=[el_ids], dims=["elements"], attrs={'units': 'meter'})
    return (
        el_ids,
        node_ids,
        time,
        xe_da,
        xn_da,
        ye_da,
        yn_da,
        z_dyn_da,
        z_dynamic,
        ze_da,
        zn_da,
    )


@app.cell
def _(
    ds,
    el_ids,
    time,
    xe_da,
    xn_da,
    xr,
    ye_da,
    yn_da,
    z_dyn_da,
    ze_da,
    zn_da,
):
    # Add coordinates for nodes and elements
    data_dict = {'x': xn_da,
                 'y' :yn_da,
                 'z' : zn_da,
                 'xe' : xe_da,
                 'ye' : ye_da,
                 'ze' : ze_da,
                 'z_dynamic' : z_dyn_da}

    # add rest of data
    for da in ds:
            da = xr.DataArray(da.to_numpy(), 
                              coords = [time,el_ids],
                              dims=["time", "elements"],
                              attrs={'units': da.unit.name})

            data_dict[da.name] = da


    # Here are some examples of global attributes, which is useful, but in most cases not required
    attributes={'title:' : "Model A.2:4",
                'history': 'mikeio | xarray',
                'source': 'Mike 3 FM - Oresund',
                'instituion': 'DHI'}

    # create an xarray dataset
    xr_ds = xr.Dataset(data_dict, attrs=attributes)
    return attributes, da, data_dict, xr_ds


@app.cell
def _(xr_ds):
    xr_ds.to_netcdf("oresund_sigma_z.nc")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Clean up
        """
    )
    return


@app.cell
def _(os):
    os.remove("oresund_sigma_z.nc")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

