import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from copy import deepcopy
from mikeio.eum import EUMType, EUMUnit, ItemInfo


def get_valid_items_and_timesteps(dfs, items, time_steps):
    # TODO consider if this should be part of a DFS base class

    if isinstance(items, int) or isinstance(items, str):
        items = [items]

    if items is not None and isinstance(items[0], str):
        items = find_item(dfs._source, items)

    if items is None:
        item_numbers = list(range(dfs._n_items))
    else:
        item_numbers = items

    if time_steps is None:
        time_steps = list(range(dfs._n_timesteps))

    if isinstance(time_steps, int):
        time_steps = [time_steps]

    if isinstance(time_steps, str):
        parts = time_steps.split(",")
        if parts[0] == "":
            time_steps = slice(parts[1])  # stop only
        elif parts[1] == "":
            time_steps = slice(parts[0], None)  # start only
        else:
            time_steps = slice(parts[0], parts[1])

    if isinstance(time_steps, slice):
        freq = pd.tseries.offsets.DateOffset(seconds=dfs.timestep)
        time = pd.date_range(dfs.start_time, periods=dfs.n_timesteps, freq=freq)
        s = time.slice_indexer(time_steps.start, time_steps.stop)
        time_steps = list(range(s.start, s.stop))

    items = get_item_info(dfs._source, item_numbers)

    return items, item_numbers, time_steps


def find_item(dfs, item_names):
    """Utility function to find item numbers

    Parameters
    ----------
    dfs : DfsFile

    item_names : list[str]
        Names of items to be found

    Returns
    -------
    list[int]
        item numbers (0-based)

    Raises
    ------
    KeyError
        In case item is not found in the dfs file
    """
    names = [x.Name for x in dfs.ItemInfo]
    item_lookup = {name: i for i, name in enumerate(names)}
    try:
        item_numbers = [item_lookup[x] for x in item_names]
    except KeyError:
        raise KeyError(f"Selected item name not found. Valid names are {names}")

    return item_numbers


def get_item_info(dfs, item_numbers):
    """Read DFS ItemInfo

    Parameters
    ----------
    dfs : MIKE dfs object
    item_numbers : list[int]
        
    Returns
    -------
    list[Iteminfo]
    """
    items = []
    for item in item_numbers:
        name = dfs.ItemInfo[item].Name
        eumItem = dfs.ItemInfo[item].Quantity.Item
        eumUnit = dfs.ItemInfo[item].Quantity.Unit
        itemtype = EUMType(eumItem)
        unit = EUMUnit(eumUnit)
        item = ItemInfo(name, itemtype, unit)
        items.append(item)
    return items

