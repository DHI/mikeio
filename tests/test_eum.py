from mikeio.eum import Item


def test_item_is_equivalent_to_int():

    assert Item.Temperature == 100006


def test_get_unit():

    assert Item.Temperature.units["degree Celsius"] == 2800
