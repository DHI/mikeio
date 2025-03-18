import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Connectivity
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, np, plt


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/oresundHD_run1.dfsu")
    ds.geometry.plot();
    return (ds,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The info on the connectivity between nodes and elements can be found in the element table
        """
    )
    return


@app.cell
def _(ds):
    et = ds.geometry.element_table
    len(et)
    return (et,)


@app.cell
def _(et):
    et[0]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's find out if any of these nodes are also found in another element, this would imply that these elements are neigbours (adjacent).
        """
    )
    return


@app.cell
def _(et):
    for _i, _e in enumerate(et):
        for _n in et[0]:
            if _n in _e:
                print(f'Node: {_n} found in element {_i}')
    return


@app.cell
def _(ds):
    ne = ds.geometry.n_elements
    return (ne,)


@app.cell
def _(ds, el, et):
    nodetable = {}
    for _el in range(ds.geometry.n_elements):
        _nodes = et[el]
        for node in _nodes:
            if node in nodetable:
                nodetable[node].append(_el)
            else:
                nodetable[node] = [el]
    return node, nodetable


@app.cell
def _():
    def is_neighbour(a, b) -> bool:
        return len(set(a).intersection(set(b))) == 2
    return (is_neighbour,)


@app.cell
def _(e, el, et, is_neighbour, n, ne, nodetable):
    ec = {}
    for _el in range(ne):
        _nodes = et[el]
        for _n in _nodes:
            elements = nodetable[n]
            for _e in elements:
                if is_neighbour(et[_el], et[_e]):
                    if _el in ec:
                        if _e not in ec[_el]:
                            ec[_el].append(_e)
                    else:
                        ec[el] = [e]
    return ec, elements


@app.cell
def _(ec):
    ec[1772]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Neighbours
        """
    )
    return


@app.cell
def _(ds):
    coords = ds.geometry.element_coordinates
    e1 = ds.geometry.find_nearest_elements(x=340000,y=6.16e6)
    e1
    return coords, e1


@app.cell
def _(e1, ec):
    e1_n = ec[e1]
    e1_n
    return (e1_n,)


@app.cell
def _(coords, ds, e1, e1_n, plt):
    _ax = ds.geometry.plot.mesh(figsize=(12, 12))
    plt.xlim(330000, 360000)
    plt.ylim(6150000.0, 6180000.0)
    plt.scatter(coords[e1, 0], coords[e1, 1], marker='*', s=200, label='Selected element')
    plt.scatter(coords[e1_n, 0], coords[e1_n, 1], marker='+', c='red', s=200, label='Neigbour elements')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Shortest path
        """
    )
    return


@app.cell
def _(ds):
    ea = ds.geometry.find_nearest_elements(x=343000,y=6168000)
    eb = ds.geometry.find_nearest_elements(x=365000,y=6168000)
    return ea, eb


@app.cell
def _(coords, ec, i, j, ne, np):
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.csgraph import shortest_path
    D = lil_matrix((ne, ne))
    for _i in range(ne):
        _row = ec[i]
        for _j in _row:
            d = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            D[i, j] = d
    D = csr_matrix(D)
    dist, pred = shortest_path(D, return_predecessors=True)
    return D, csr_matrix, d, dist, lil_matrix, pred, shortest_path


@app.cell
def _(dist, ea, eb):
    dist[ea,eb]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The predessors matrix `pred` encodes the previous step in the shortest path between any node (in this respect a node in the graph is an element) in the graph.
        In order to get all steps in the path between two elements we can loop through the steps.
        """
    )
    return


@app.cell
def _(ea, eb, n, pred):
    path = [eb]
    _n = eb
    while _n != ea:
        _n = pred[ea, n]
        path.append(_n)
    path[0:10]
    return (path,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The path between two elements is here to illustrate how the distance along the shortest path is calculated, you don't need to use the `pred` matrix if you are only interested in the distance.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Calculate the distance through air (ignoring land).
        """
    )
    return


@app.cell
def _(coords, ea, eb, np):
    euc_dist = np.sqrt(np.sum((coords[ea,:2] - coords[eb,:2])**2))
    return (euc_dist,)


@app.cell
def _(coords, dist, ds, ea, eb, euc_dist, path, plt):
    _ax = ds.geometry.plot.mesh(figsize=(12, 12), title=f'Distance through air: {euc_dist / 1000:.0f} km\nDistance through water: {dist[ea, eb] / 1000:.0f} km')
    plt.xlim(330000, 370000)
    plt.ylim(6150000.0, 6180000.0)
    plt.scatter(coords[ea, 0], coords[ea, 1], marker='*', s=200, label='Element A')
    plt.scatter(coords[eb, 0], coords[eb, 1], marker='*', s=200, label='Element B')
    plt.scatter(coords[path, 0], coords[path, 1], marker='.', c='green', s=100, label='Shortest path')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clustering
        """
    )
    return


@app.cell
def _(ec, i, j, lil_matrix, ne):
    C = lil_matrix((ne, ne))
    for _i in range(ne):
        _row = ec[i]
        for _j in _row:
            C[i, j] = 1
    return (C,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds):
    data = ds.Surface_elevation.values.T
    data.shape
    return (data,)


@app.cell
def _(C, data):
    from sklearn.cluster import AgglomerativeClustering

    c = AgglomerativeClustering(
                n_clusters=10, connectivity=C, linkage="ward"
        ).fit(data)
    return AgglomerativeClustering, c


@app.cell
def _(c):
    c.labels_
    return


@app.cell
def _(c, ds, mikeio):
    da = mikeio.DataArray(c.labels_, geometry=ds.geometry, item="Cluster #")
    da
    return (da,)


@app.cell
def _(da):
    da.plot(figsize=(12,12), cmap='tab10')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

