import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Mesh""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, plt


@app.cell
def _(mikeio):
    msh = mikeio.Mesh("../tests/testdata/odense_rough.mesh")
    msh
    return (msh,)


@app.cell
def _(msh):
    msh.plot()
    msh.plot.boundary_nodes(boundary_names=['Land','Open boundary'])
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
    mo.md(r"""Find if points are inside the domain""")
    return


@app.cell
def _(mp):
    from shapely.geometry import Point

    p1 = Point(216000, 6162000)
    p2 = Point(220000, 6156000)
    [
        mp.contains(p1),
        mp.contains(p2)
    ]
    return Point, p1, p2


@app.cell
def _(msh):
    p1p2 = [[216000, 6162000], [220000, 6156000]]
    msh.geometry.contains(p1p2)
    return (p1p2,)


@app.cell
def _(msh, p1, p2):
    ax = msh.plot(title="", levels=range(-10,1))
    ax.scatter(p1.x, p1.y, marker="*", s=200, c="red", label="inside")
    ax.scatter(p2.x, p2.y, marker="+", s=200, c="green", label="outside")
    ax.legend()
    return (ax,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
