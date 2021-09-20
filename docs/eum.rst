.. _eum:

EUM and items
*************

The dfs items in MIKE IO are represented by the `ItemInfo class <eum.html#mikeio.eum.ItemInfo>`_. 
An ItemInfo consists of:

* name - a user-defined string 
* type - an `EUMType <eum.html#mikeio.eum.EUMType>`_ 
* unit - an `EUMUnit <eum.html#mikeio.eum.EUMUnit>`_

.. code-block:: python

    >>> from mikeio.eum import ItemInfo, EUMType
    >>> item = ItemInfo("Viken", EUMType.Water_Level)
    >>> item
    Viken <Water Level> (meter)
    >>> ItemInfo(EUMType.Wind_speed)
    Wind speed <Wind speed> (meter per sec)

See the `Units notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Units.ipynb>`_ for more examples.




EUM API
-------
.. automodule:: mikeio.eum
	:members:
