from mikeio.eum import Item


def test_item_is_equivalent_to_int():

    assert Item.Temperature == 100006


def test_item_code():

    assert Item.Temperature.code == 100006


def test_get_unit():

    assert len(Item.Temperature.units) == 3


def test_get_item_name():

    assert Item.Water_Level.display_name == "Water Level"


def test_get_item_repr():

    assert repr(Item.Water_Level) == "Water Level"
