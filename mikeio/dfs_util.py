from DHI.Generic.MikeZero import EUMWrapper
import numpy as np


def type_list(search=None):
    """Get a dictionary of the EUM items

    Notes
    -----
    An alternative to `type_list` is to use `mikeio.eum.Item`

    Parameters
    ----------
    search: str, optional
        a search string (caseinsensitive) to filter out results

    Returns
    -------
    dict
        names and codes for EUM items
    """
    items = {}
    check = True
    i = 1
    while check:
        d = EUMWrapper.eumGetItemTypeSeq(i, 0, "")
        if d[0] is True:
            items[d[1]] = d[2]
            i += 1
        else:
            check = False

    if search is not None:
        search = search.lower()
        items = dict(
            [
                [key, value]
                for key, value in items.items()
                if search in value.lower() or search == value.lower()
            ]
        )

    return items


def unit_list(type_enum, search=None):
    """Get a dictionary of valid units

    Parameters
    ----------
    type_enum: int
        EUM variable type, e.g. 100006 or Item.Temperature

    Returns
    -------
    dict
        names and codes for valid units
    """
    items = {}
    for i in range(EUMWrapper.eumGetItemUnitCount(type_enum)):
        d = EUMWrapper.eumGetItemUnitSeq(type_enum, i + 1, 1, "")
        if d[0] is True:
            items[d[2]] = d[1]

    if search is not None:
        items = dict(
            [
                [key, value]
                for key, value in items.items()
                if search.lower() in key.lower() or search.lower() == key.lower()
            ]
        )

    return items


def timestep_list():
    """Produces a dictionary of the possible time units. For example: seconds, minutes, days, ...
    """
    item = type_list(search="timestep")
    key = list(item.keys())[0]
    items = unit_list(key)
    return items


def grid_centers_from_coordinates(X0, Y0, nx, ny, dx, dy):
    """Producses a grid of x, y values of the center of a grid, based on the
        provided X0, Y0, nx, ny, dx, dy
    """
    x = np.zeros(shape=(ny, nx))
    y = np.zeros(shape=(ny, nx))

    xvalues = np.arange(X0 + dx / 2, X0 + dx / 2 + nx * dx, step=dx)
    yvalues = np.arange(Y0 - dy / 2 + ny * dy, Y0 - dy / 2, step=-dy)

    for i in range(ny):
        x[i, :] = xvalues
    for i in range(nx):
        y[:, i] = yvalues

    return x, y
