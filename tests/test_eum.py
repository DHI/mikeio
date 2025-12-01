import pytest
from mikeio import EUMType, EUMUnit, ItemInfo
from mikeio.eum import ItemInfoList

from mikecore.eum import eumItem, eumUnit


def test_item_is_equivalent_to_int() -> None:
    assert EUMType.Temperature == 100006


def test_item_code() -> None:
    assert EUMType.Temperature.code == 100006


def test_get_unit() -> None:
    assert len(EUMType.Temperature.units) == 3
    assert EUMType.Temperature.units[0] == EUMUnit.degree_Celsius


def test_get_item_name() -> None:
    assert EUMType.Water_Level.display_name == "Water Level"


def test_get_item_repr() -> None:
    assert repr(EUMType.Water_Level) == "Water Level"


def test_create_info_with_name() -> None:
    item = ItemInfo("Foo")

    assert item.name == "Foo"


def test_create_info_with_type_only() -> None:
    item = ItemInfo(itemtype=EUMType.Water_Level)

    assert item.name == "Water Level"


def test_create_info_with_type_only_positional() -> None:
    item = ItemInfo(EUMType.Water_Level)

    assert item.name == "Water Level"
    assert item.type == EUMType.Water_Level


def test_equality() -> None:
    item1 = ItemInfo("Foo", EUMType.Water_Level)
    item2 = ItemInfo("Foo", EUMType.Water_Level)

    assert item1 == item2


def test_eum_type_search() -> None:
    types = EUMType.search("velocity")

    assert len(types) > 0
    assert EUMType.Wind_Velocity in types


def test_eum_conversion() -> None:
    """Verify that all EUM types and units in mikecore have equivalents in MIKE IO"""

    for code in eumItem:
        EUMType(code)

    for code in eumUnit:
        EUMUnit(code)

    assert True


def test_short_name() -> None:
    assert EUMType.Water_Level.units[0].name == "meter"
    assert EUMType.Water_Level.units[0].short_name == "m"

    assert EUMType.Acceleration.units[0].name == "meter_per_sec_pow_2"
    assert EUMType.Acceleration.units[0].short_name == "m/s^2"

    assert EUMUnit.gallonUK.name == "gallonUK"
    assert EUMUnit.gallonUK.short_name == "gallonUK"


def test_item_info_list() -> None:
    items = [ItemInfo("Foo", EUMType.Water_Level), ItemInfo("Bar", EUMType.Temperature)]

    itemlist = ItemInfoList(items)

    assert itemlist[0].name == "Foo"
    df = itemlist.to_dataframe()
    assert df["name"][0] == "Foo"


def test_default_type() -> None:
    item = ItemInfo("Foo")
    assert item.type == EUMType.Undefined
    assert repr(item.unit) == "undefined"


def test_int_is_valid_type_info() -> None:
    item = ItemInfo("Foo", 100123)
    assert item.type == EUMType.Viscosity

    item = ItemInfo("U", 100002)
    assert item.type == EUMType.Wind_Velocity


def test_int_is_valid_unit_info() -> None:
    item = ItemInfo("U", 100002, 2000)
    assert item.type == EUMType.Wind_Velocity
    assert item.unit == EUMUnit.meter_per_sec
    assert repr(item.unit) == "meter per sec"  # TODO replace _per_ with /


def test_default_unit_from_type() -> None:
    item = ItemInfo("Foo", EUMType.Water_Level)
    assert item.type == EUMType.Water_Level
    assert item.unit == EUMUnit.meter
    assert repr(item.unit) == "meter"

    item = ItemInfo("Tp", EUMType.Wave_period)
    assert item.type == EUMType.Wave_period
    assert item.unit == EUMUnit.second
    assert repr(item.unit) == "second"

    item = ItemInfo("Temperature", EUMType.Temperature)
    assert item.type == EUMType.Temperature
    assert item.unit == EUMUnit.degree_Celsius
    assert repr(item.unit) == "degree Celsius"


def test_default_name_from_type() -> None:
    item = ItemInfo(EUMType.Current_Speed)
    assert item.name == "Current Speed"
    assert item.unit == EUMUnit.meter_per_sec

    item2 = ItemInfo(EUMType.Current_Direction, EUMUnit.degree)
    assert item2.unit == EUMUnit.degree
    item3 = ItemInfo(
        "Current direction (going to)", EUMType.Current_Direction, EUMUnit.degree
    )
    assert item3.type == EUMType.Current_Direction
    assert item3.unit == EUMUnit.degree


def test_iteminfo_string_type_should_fail_with_helpful_message() -> None:
    with pytest.raises(ValueError):
        ItemInfo("Water level", "Water level")  # type: ignore
