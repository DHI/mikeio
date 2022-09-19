# EUM

The dfs items in MIKE IO are represented by the `ItemInfo` class. 
An ItemInfo consists of:

* name - a user-defined string 
* type - an `EUMType` 
* unit - an `EUMUnit`

The ItemInfo class has some sensible defaults, thus you can specify only a name or a type. If you don't specify a unit, the default unit for that type will be used.

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

Matching units for specific type:
```python
>>> EUMType.Wind_speed.units
[meter per sec, feet per sec, knot, km per hour, miles per hour]
```

Default unit:
```python
>>> EUMType.Precipitation_Rate.units[0]
mm per day
>>> unit = EUMType.Precipitation_Rate.units[0]
>>> unit
mm per day
>>> type(unit)
<enum 'EUMUnit'>
>>> int(unit)
2004
```

## EUM type search

It is also possible to do a string based search, e.g. to find all EUM types containing the substring 'period':

```python
>>> EUMType.search("period")
[Wave period, Return period, Update Period, Threshold period]
```


## EUM API

```{eval-rst}
.. automodule:: mikeio.eum
	:members:
```
