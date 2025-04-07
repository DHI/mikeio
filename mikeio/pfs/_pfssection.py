from __future__ import annotations
from collections.abc import KeysView, ValuesView
from datetime import datetime
from types import SimpleNamespace
from typing import (
    Any,
    ItemsView,
    Mapping,
    MutableMapping,
    Sequence,
)

import pandas as pd


def _merge_dict(a: dict[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """merges dict b into dict a; handling non-unique keys."""
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dict(a[key], b[key])
            else:
                ab = list(a[key]) + list(b[key])
                a[key] = PfsNonUniqueList(ab)
        else:
            a[key] = b[key]
    return a


class PfsNonUniqueList(list):
    # TODO do we really need this, regular lists are not unique
    pass


class PfsSection(SimpleNamespace, MutableMapping[str, Any]):
    """Class for reading/writing sections in a pfs file."""

    @staticmethod
    def from_dataframe(df: pd.DataFrame, prefix: str) -> "PfsSection":
        """Create a PfsSection from a DataFrame.

        Parameters
        ----------
        df: dataframe
            data
        prefix: str
            section header prefix

        Examples
        --------
        ```{python}
        import pandas as pd
        import mikeio
        df = pd.DataFrame(dict(station=["Foo", "Bar"],include=[0,1]), index=[1,2])
        df
        ```

        ```{python}
        mikeio.PfsSection.from_dataframe(df,"STATION_")
        ```

        """
        d = {f"{prefix}{idx}": row.to_dict() for idx, row in df.iterrows()}

        return PfsSection(d)

    def __init__(self, dictionary: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            self.__set_key_value(key, value, copy=True)

    def __repr__(self) -> str:
        return "\n".join(self._to_txt_lines())

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> Any:
        iter(self)

    def __contains__(self, key: Any) -> bool:
        return key in self.keys()

    def __getitem__(self, key: str) -> Any:
        SECTION_SEPARATOR = "/"
        subsections = key.split(SECTION_SEPARATOR)
        item = getattr(self, subsections[0])
        for section in subsections[1:]:
            item = getattr(item, section)
        return item

    def __setitem__(self, key: str, value: Any) -> None:
        SECTION_SEPARATOR = "/"
        subsections = key.split(SECTION_SEPARATOR)
        current_section = self
        for section in subsections[:-1]:
            if not hasattr(current_section, section):
                setattr(current_section, section, PfsSection({}))
            current_section = getattr(current_section, section)
        current_section.__set_key_value(subsections[-1], value)

    def __delitem__(self, key: str) -> None:
        if key in self.keys():
            self.__delattr__(key)
        else:
            raise IndexError("Key not found")

    def __set_key_value(self, key: str, value: Any, copy: bool = False) -> None:
        if value is None:
            value = {}

        match value:
            case dict():
                d = value.copy() if copy else value
                self.__setattr__(key, PfsSection(d))
            case PfsNonUniqueList():
                # multiple keywords/Sections with same name
                sections = PfsNonUniqueList()
                for v in value:
                    match v:
                        case dict():
                            d = v.copy() if copy else v
                            sections.append(PfsSection(d))
                        case _:
                            sections.append(self._parse_value(v))
                self.__setattr__(key, sections)
            case _:
                self.__setattr__(key, self._parse_value(value))

    def _parse_value(self, v: Any) -> Any:
        if isinstance(v, str) and self._str_is_scientific_float(v):
            return float(v)
        return v

    @staticmethod
    def _str_is_scientific_float(s: str) -> bool:
        """True: -1.0e2, 1E-4, -0.1E+0.5; False: E12, E-4."""
        if len(s) < 3 or s.lower().startswith("e"):
            return False
        try:
            float(s)
            return "e" in s.lower()
        except ValueError:
            return False

    def pop(self, key: Any, default: Any = None) -> Any:
        """If key is in the dictionary, remove it and return its
        value, else return default. If default is not given and
        key is not in the dictionary, a KeyError is raised."""
        return self.__dict__.pop(key, default)

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for key if key is in the PfsSection,
        else default. If default is not given, it defaults to None,
        so that this method never raises a KeyError."""
        return self.__dict__.get(key, default)

    def clear(self) -> None:
        """Remove all items from the PfsSection."""
        return self.__dict__.clear()

    def keys(self) -> KeysView[str]:
        """Return a new view of the PfsSection's keys."""
        return self.__dict__.keys()

    def values(self) -> ValuesView[Any]:
        """Return a new view of the PfsSection's values."""
        return self.__dict__.values()

    def items(self) -> ItemsView[str, Any]:
        """Return a new view of the PfsSection's items ((key, value) pairs)."""
        return self.__dict__.items()

    def search(
        self,
        text: str | None = None,
        *,
        key: str | None = None,
        section: str | None = None,
        param: str | bool | int | float | None = None,
        case: bool = False,
    ) -> PfsSection:
        """Find recursively all keys, sections or parameters
           matching a pattern.

        NOTE: logical OR between multiple conditions

        Parameters
        ----------
        text : str, optional
            Search for text in either key, section or parameter, by default None
        key : str, optional
            text pattern to seach for in keywords, by default None
        section : str, optional
            text pattern to seach for in sections, by default None
        param : str, bool, float, int, optional
            text or value in a parameter, by default None
        case : bool, optional
            should the text search be case-sensitive?, by default False

        Returns
        -------
        PfsSection
            Search result as a nested PfsSection

        """
        if text is not None:
            # text searches across all fields
            if key is not None or section is not None or param is not None:
                raise ValueError(
                    "When 'text' is provided, 'key', 'section' and 'param' must be None"
                )
            key = section = param = text

        key = key.lower() if (key is not None and not case) else key
        section = section.lower() if (section is not None and not case) else section
        param = (
            param
            if (param is None or not isinstance(param, str) or case)
            else param.lower()
        )
        results = [
            item
            for item in self._find_patterns_generator(
                keypat=key, parampat=param, secpat=section, case=case
            )
        ]
        return (
            self.__class__._merge_PfsSections(results)
            if len(results) > 0
            else PfsSection({})
        )

    def _find_patterns_generator(
        self,
        keypat: str | None = None,
        parampat: Any = None,
        secpat: str | None = None,
        keylist: list[str] | None = None,
        case: bool = False,
    ) -> Any:
        """Look for patterns in either keys, params or sections."""
        keylist = [] if keylist is None else keylist
        for k, v in self.items():
            kk = str(k) if case else str(k).lower()

            if isinstance(v, PfsSection):
                if secpat and secpat in kk:
                    yield from self._yield_deep_dict(keylist + [k], v)
                else:
                    yield from v._find_patterns_generator(
                        keypat, parampat, secpat, keylist=keylist + [k], case=case
                    )
            else:
                if (keypat and keypat in kk) or self._param_match(parampat, v, case):
                    yield from self._yield_deep_dict(keylist + [k], v)

    @staticmethod
    def _yield_deep_dict(keys: Sequence[str], val: Any) -> Any:
        """yield a deep nested dict with keys with a single deep value val."""
        for j in range(len(keys) - 1, -1, -1):
            d = {keys[j]: val}
            val = d
        yield d

    @staticmethod
    def _param_match(parampat: Any, v: Any, case: bool) -> Any:
        if parampat is None:
            return False
        if type(v) is not type(parampat):
            return False
        if isinstance(v, str):
            vv = str(v) if case else str(v).lower()
            return parampat in vv
        else:
            return parampat == v

    def find_replace(self, old_value: Any, new_value: Any) -> None:
        """Update recursively all old_value with new_value."""
        for k, v in self.items():
            if isinstance(v, PfsSection):
                self[k].find_replace(old_value, new_value)
            elif self[k] == old_value:
                self[k] = new_value

    def copy(self) -> "PfsSection":
        """Return a copy of the PfsSection."""
        return PfsSection(self.to_dict())

    def _to_txt_lines(self) -> list[str]:
        lines: list[str] = []
        self._append_to_lines_at_level(lines)
        return lines

    def _append_to_lines_at_level(self, lines: list[str], level: int = 0) -> None:
        lvl_prefix = "   " * level
        for k, v in self.items():
            # check for empty sections
            if v is None:
                lines.append(f"{lvl_prefix}[{k}]")
                lines.append(f"{lvl_prefix}EndSect  // {k}")

            elif isinstance(v, list) and any(
                isinstance(subv, PfsSection) for subv in v
            ):
                # duplicate sections
                for subv in v:
                    if isinstance(subv, PfsSection):
                        subsec = PfsSection({k: subv})
                        subsec._append_to_lines_at_level(lines, level=level)
                    else:
                        subv = self._prepare_value_for_write(subv)
                        lines.append(f"{lvl_prefix}{k} = {subv}")
            elif isinstance(v, PfsSection):
                lines.append(f"{lvl_prefix}[{k}]")
                v._append_to_lines_at_level(lines, level=(level + 1))
                lines.append(f"{lvl_prefix}EndSect  // {k}")
            elif isinstance(v, PfsNonUniqueList) or (
                isinstance(v, list) and all([isinstance(vv, list) for vv in v])
            ):
                if len(v) == 0:
                    # empty list -> keyword with no parameter
                    lines.append(f"{lvl_prefix}{k} = ")
                for subv in v:
                    psubv = self._prepare_value_for_write(subv)
                    lines.append(f"{lvl_prefix}{k} = {psubv}")
            else:
                pv = self._prepare_value_for_write(v)
                lines.append(f"{lvl_prefix}{k} = {pv}")

    def _prepare_value_for_write(
        self, v: str | bool | datetime | list[str | bool | datetime]
    ) -> str:
        """Catch peculiarities of string formatted pfs data."""
        match v:
            case str():
                if len(v) > 5 and not ("PROJ" in v or "<CLOB:" in v):
                    v = v.replace('"', "''")
                    v = v.replace("\U0001f600", "'")

                if v == "":
                    v = "''"
                elif v.count("|") == 2 and "CLOB" not in v:
                    v = f"{v}"
                else:
                    v = f"'{v}'"

            case bool():
                v = str(v).lower()  # stick to MIKE lowercase bool notation

            case datetime():
                v = v.strftime("%Y, %m, %d, %H, %M, %S").replace(" 0", " ")

            case list():
                out = []
                for subv in v:
                    out.append(str(self._prepare_value_for_write(subv)))
                v = ", ".join(out)

        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to (nested) dict (as a copy)."""
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, PfsSection):
                d[key] = value.to_dict()
        return d

    def to_dataframe(self, prefix: str | None = None) -> pd.DataFrame:
        """Output enumerated subsections to a DataFrame.

        Parameters
        ----------
        prefix : str, optional
            The prefix of the enumerated sections, e.g. "OUTPUT_",
            which can be supplied if it fails without this argument,
            by default None (will try to "guess" the prefix)

        Returns
        -------
        pd.DataFrame
            The enumerated subsections as a DataFrame

        Examples
        --------
        ```{python}
        pfs = mikeio.read_pfs("../data/pfs/lake.sw")
        pfs.SW.OUTPUTS.to_dataframe(prefix="OUTPUT_")
        ```

        """
        if prefix is not None:
            sections = [
                k for k in self.keys() if k.startswith(prefix) and k[-1].isdigit()
            ]
            n_sections = len(sections)
        else:
            n_sections = -1
            # TODO: check that value is a PfsSection
            sections = [k for k in self.keys() if k[-1].isdigit()]
            for k in self.keys():
                if isinstance(k, str) and k.startswith("number_of_"):
                    n_sections = self[k]
            if n_sections == -1:
                # raise ValueError("Could not find a number_of_... keyword")
                n_sections = len(sections)

        if len(sections) == 0:
            prefix_txt = "" if prefix is None else f"(starting with '{prefix}') "
            raise ValueError(f"No enumerated subsections {prefix_txt}found")

        prefix = sections[0][:-1]
        res = []
        for j in range(n_sections):
            k = f"{prefix}{j + 1}"
            res.append(self[k].to_dict())
        return pd.DataFrame(res, index=range(1, n_sections + 1))

    @classmethod
    def _merge_PfsSections(cls, sections: Sequence[dict[str, Any]]) -> "PfsSection":
        """Merge a list of PfsSections/dict."""
        assert len(sections) > 0
        a = sections[0]
        for b in sections[1:]:
            a = _merge_dict(a, b)
        return cls(a)
