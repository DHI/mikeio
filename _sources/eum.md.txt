# EUM and items

The dfs items in MIKE IO are represented by the `ItemInfo` class. 
An ItemInfo consists of:

* name - a user-defined string 
* type - an `EUMType` 
* unit - an `EUMUnit`

```python
>>> from mikeio import ItemInfo, EUMType
>>> item = ItemInfo("Viken", EUMType.Water_Level)
>>> item
Viken <Water Level> (meter)
>>> ItemInfo(EUMType.Wind_speed)
Wind speed <Wind speed> (meter per sec)
>>> ItemInfo("Viken", EUMType.Water_Level, EUMUnit.feet)
Viken <Water Level> (feet)
```

See the [Units notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Units.ipynb) for more examples.


EUM API
-------
```{eval-rst}
.. automodule:: mikeio.eum
	:members:
```
