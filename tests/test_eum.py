from mikeio.eum import EUMType, EUMUnit, ItemInfo

from mikecore.eum import eumItem, eumUnit


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


def test_equality():

    item1 = ItemInfo("Foo", EUMType.Water_Level)
    item2 = ItemInfo("Foo", EUMType.Water_Level)

    assert item1 == item2


def test_eum_type_search():

    types = EUMType.search("velocity")

    assert len(types) > 0


def test_eum_conversion():
    """Verify that all EUM types and units in mikecore have equivalents in MIKE IO"""

    for code in eumItem:
        EUMType(code)

    for code in eumUnit:
        EUMUnit(code)

    assert True


def test_short_name():
    assert EUMType.Water_Level.units[0].name == "meter"
    assert EUMType.Water_Level.units[0].short_name == "m"

    assert EUMType.Acceleration.units[0].name == "meter_per_sec_pow_2"
    assert EUMType.Acceleration.units[0].short_name == "m/s^2"

    assert EUMUnit.gallonUK.name == "gallonUK"
    assert EUMUnit.gallonUK.short_name == "gallonUK"
