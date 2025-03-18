import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Mesh
        * read mesh file
        * plot mesh 
        * convert to shapely
        * check if point is inside or outside mesh
        * subset mesh, plot subset
        * change z values
        * change boundary codes
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('png')
    plt.rcParams["figure.figsize"] = (6,6)

    import mikeio
    return mikeio, plt, set_matplotlib_formats


@app.cell
def _(mikeio):
    msh = mikeio.Mesh("../tests/testdata/odense_rough.mesh")
    msh
    return (msh,)


@app.cell
def _(msh):
    msh.plot()
    msh.plot.boundary_nodes(boundary_names=['Land','Open boundary']);
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Convert mesh to shapely
        Convert mesh to [shapely](https://shapely.readthedocs.io/en/latest/manual.html) MultiPolygon object, requires that the `shapely` library is installed.
        """
    )
    return


@app.cell
def _(msh):
    mp = msh.to_shapely()
    mp
    return (mp,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Now a lot of methods are available
        """
    )
    return


@app.cell
def _(mp):
    mp.area
    return


@app.cell
def _(mp):
    mp.bounds
    return


@app.cell
def _(mp):
    domain = mp.buffer(0)
    domain
    return (domain,)


@app.cell
def _(domain):
    open_water = domain.buffer(-500)

    coastalzone = domain - open_water
    coastalzone
    return coastalzone, open_water


@app.cell
def _(mo):
    mo.md(
        r"""
        Find if points are inside the domain
        """
    )
    return


@app.cell
def _(mp):
    from shapely.geometry import Point

    p1 = Point(216000, 6162000)
    p2 = Point(220000, 6156000)
    print(mp.contains(p1))
    print(mp.contains(p2))
    return Point, p1, p2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Mesh class can also check if a mesh contains points 
        """
    )
    return


@app.cell
def _(msh):
    p1p2 = [[216000, 6162000], [220000, 6156000]]
    msh.geometry.contains(p1p2)
    return (p1p2,)


@app.cell
def _(msh, p1, p2):
    ax = msh.plot()
    ax.scatter(p1.x, p1.y, marker="*", s=200, c="red", label="inside")
    ax.scatter(p2.x, p2.y, marker="+", s=200, c="green", label="outside")
    ax.legend();
    return (ax,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Subset mesh
        Select only elements with more than 3m depth. Plot these elements. 
        """
    )
    return


@app.cell
def _(msh):
    _zc = msh.element_coordinates[:, 2]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Change z values and boundary code
        Assume that we want to have a minimum depth of 2 meters and change the open boundary (code 2) to a closed one (code 1). 
        """
    )
    return


@app.cell
def _(msh, zc):
    print(f'max z before: {msh.node_coordinates[:, 2].max()}')
    _zc = msh.node_coordinates[:, 2]
    zc[zc > -2] = -2
    msh.zn = zc
    print(f'max z after: {msh.node_coordinates[:, 2].max()}')
    return


@app.cell
def _(msh):
    c = msh.geometry.codes
    c[c==2] = 1
    msh.geometry.codes = c
    return (c,)


@app.cell
def _(msh):
    msh.geometry.codes
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

