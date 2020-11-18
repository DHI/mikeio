from mikeio.eum import EUMType, ItemInfo


def test_item_is_equivalent_to_int():

    assert EUMType.Temperature == 100006


def test_item_code():

    assert EUMType.Temperature.code == 100006


def test_get_unit():

    assert len(EUMType.Temperature.units) == 3


def test_get_item_name():

    assert EUMType.Water_Level.display_name == "Water Level"


def test_get_item_repr():

    assert repr(EUMType.Water_Level) == "Water Level"


def test_create_info_with_name():

    item = ItemInfo("Foo")

    assert item.name == "Foo"


def test_create_info_with_type_only():

    item = ItemInfo(itemtype=EUMType.Water_Level)

    assert item.name == "Water Level"


def test_create_info_with_type_only_positional():

    item = ItemInfo(EUMType.Water_Level)

    assert item.name == "Water Level"
    assert item.type == EUMType.Water_Level
