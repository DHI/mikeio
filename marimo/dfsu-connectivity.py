import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Dfsu - Connectivity""")
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
    mo.md(r"""The info on the connectivity between nodes and elements can be found in the element table""")
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
    mo.md(r"""Let's find out if any of these nodes are also found in another element, this would imply that these elements are neigbours (adjacent).""")
    return


@app.cell
def _(et):
    def _():
        for i, e in enumerate(et):
            for n in et[0]:
                if n in e:
                    print(f'Node: {n} found in element {i}')

    _()
    return


@app.cell
def _(ds):
    ne = ds.geometry.n_elements
    return (ne,)


@app.cell
def _(ds, et):
    nodetable = {}
    for el in range(ds.geometry.n_elements):
        nodes = et[el]
        for node in nodes:
            if node in nodetable:
                nodetable[node].append(el)
            else:
                nodetable[node] = [el]
    return el, node, nodes, nodetable


@app.cell
def is_neighbour():
    def is_neighbour(a, b) -> bool:
        return len(set(a).intersection(set(b))) == 2
    return (is_neighbour,)


@app.cell
def _(et, is_neighbour, ne, nodetable):
    def find_neighbours():
        ec = {}
        for el in range(ne):
            nodes = et[el]
            for n in nodes:
                elements = nodetable[n]
                for e in elements:
                    if is_neighbour(et[el], et[e]):
                        if el in ec:
                            if e not in ec[el]:
                                ec[el].append(e)
                        else:
                            ec[el] = [e]
        return ec
    ec = find_neighbours()
    return ec, find_neighbours


@app.cell
def _(ec):
    ec[1772]
    return


@app.cell
def _(mo):
    mo.md(r"""## Neighbours""")
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
    mo.md(r"""## Shortest path""")
    return


@app.cell
def _(ds):
    ea = ds.geometry.find_nearest_elements(x=343000,y=6168000)
    eb = ds.geometry.find_nearest_elements(x=365000,y=6168000)
    return ea, eb


@app.cell
def _(coords, ec, ne, np):
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.csgraph import shortest_path
    D = lil_matrix((ne, ne))
    for i in range(ne):
        row = ec[i]
        for j in row:
            d = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            D[i, j] = d
    D = csr_matrix(D)
    dist, pred = shortest_path(D, return_predecessors=True)
    return D, csr_matrix, d, dist, i, j, lil_matrix, pred, row, shortest_path


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
def _(ea, eb, pred):
    path = [eb]
    n = eb
    while n != ea:
        n = pred[ea, n]
        path.append(n)
    path[0:10]
    return n, path


@app.cell
def _(mo):
    mo.md(r"""The path between two elements is here to illustrate how the distance along the shortest path is calculated, you don't need to use the `pred` matrix if you are only interested in the distance.""")
    return


@app.cell
def _(mo):
    mo.md(r"""Calculate the distance through air (ignoring land).""")
    return


@app.cell
def _(coords, ea, eb, np):
    euc_dist = np.sqrt(np.sum((coords[ea,:2] - coords[eb,:2])**2))
    return (euc_dist,)


@app.cell
def _(coords, dist, ds, ea, eb, euc_dist, path, plt):
    _ax = ds.geometry.plot.mesh(figsize=(12, 12), title=f'Distance through air: {euc_dist / 1000:.0f} km\nDistance through element centers: {dist[ea, eb] / 1000:.0f} km')
    plt.xlim(330000, 370000)
    plt.ylim(6150000.0, 6180000.0)
    plt.scatter(coords[ea, 0], coords[ea, 1], marker='*', s=200, label='Element A')
    plt.scatter(coords[eb, 0], coords[eb, 1], marker='*', s=200, label='Element B')
    plt.scatter(coords[path, 0], coords[path, 1], marker='.', c='green', s=100, label='Shortest path')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""## Clustering""")
    return


@app.cell
def _(ec, lil_matrix, ne):

    def connectity_matrix():
        C = lil_matrix((ne, ne))
        for i in range(ne):
            row = ec[i]
            for j in row:
                C[i, j] = 1
        return C
    C = connectity_matrix()
    return C, connectity_matrix


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
def _(C):
    C
    return


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
