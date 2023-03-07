from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd


def _merge_dict(a: MutableMapping[str, Any], b: Mapping[str, Any]) -> Mapping[str, Any]:
    """merges dict b into dict a; handling non-unique keys"""
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
    pass


class PfsSection(SimpleNamespace, MutableMapping):

    @staticmethod
    def from_dataframe(df: pd.DataFrame, prefix: str) -> "PfsSection":
        """Create a PfsSection from a DataFrame
        
        Parameters
        ----------
        df: dataframe
            data
        prefix: str
            section header prefix
        
        Examples
        --------
        >>> df = pd.DataFrame(dict(station=["Foo", "Bar"],include=[0,1]), index=[1,2])
        >>> df
          station  include
        1     Foo        0
        2     Bar        1
        >>> mikeio.PfsSection.from_dataframe(df,"STATION_")
        ... # doctest: +NORMALIZE_WHITESPACE
        [STATION_1]
            station = 'Foo'
            include = 0
        EndSect  // STATION_1
        [STATION_2]
            station = 'Bar'
            include = 1
        EndSect  // STATION_2
        """
        d = {}
        for idx in df.index:
            key = prefix + str(idx)
            value = df.loc[idx].to_dict()
            d[key] = value
        return PfsSection(d)

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            self.__set_key_value(key, value, copy=True)

    def __repr__(self) -> str:
        return "\n".join(self._to_txt_lines())

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        iter(self)

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__set_key_value(key, value)

    def __delitem__(self, key):
        if key in self.keys():
            self.__delattr__(key)
        else:
            raise IndexError("Key not found")

    def __set_key_value(self, key, value, copy=False):
        if value is None:
            value = {}

        if isinstance(value, dict):
            d = value.copy() if copy else value
            self.__setattr__(key, PfsSection(d))
        elif isinstance(value, PfsNonUniqueList):
            # multiple keywords/Sections with same name
            sections = PfsNonUniqueList()
            for v in value:
                if isinstance(v, dict):
                    d = v.copy() if copy else v
                    sections.append(PfsSection(d))
                else:
                    sections.append(self._parse_value(v))
            self.__setattr__(key, sections)
        else:
            self.__setattr__(key, self._parse_value(value))

    def _parse_value(self, v):
        if isinstance(v, str) and self._str_is_scientific_float(v):
            return float(v)
        return v

    @staticmethod
    def _str_is_scientific_float(s):
        """True: -1.0e2, 1E-4, -0.1E+0.5; False: E12, E-4"""
        if len(s) < 3:
            return False
        if (
            s.count(".") <= 2
            and s.lower().count("e") == 1
            and s.lower()[0] != "e"
            and s.strip()
            .lower()
            .replace(".", "")
            .replace("e", "")
            .replace("-", "")
            .replace("+", "")
            .isnumeric()
        ):
            try:
                float(s)
                return True
            except:
                return False
        else:
            return False

    def pop(self, key, *args):
        """If key is in the dictionary, remove it and return its
        value, else return default. If default is not given and
        key is not in the dictionary, a KeyError is raised."""
        return self.__dict__.pop(key, *args)

    def get(self, key, *args):
        """Return the value for key if key is in the PfsSection,
        else default. If default is not given, it defaults to None,
        so that this method never raises a KeyError."""
        return self.__dict__.get(key, *args)

    def clear(self):
        """Remove all items from the PfsSection."""
        return self.__dict__.clear()

    def keys(self):
        """Return a new view of the PfsSection's keys"""
        return self.__dict__.keys()

    def values(self):
        """Return a new view of the PfsSection's values."""
        return self.__dict__.values()

    def items(self):
        """Return a new view of the PfsSection's items ((key, value) pairs)"""
        return self.__dict__.items()

    # TODO: better name
    def update_recursive(self, key, value):
        """Update recursively all matches of key with value"""
        for k, v in self.items():
            if isinstance(v, PfsSection):
                self[k].update_recursive(key, value)
            elif k == key:
                self[k] = value

    def search(
        self,
        text: Optional[str] = None,
        *,
        key: Optional[str] = None,
        section: Optional[str] = None,
        param=None,
        case: bool = False,
    ):
        """Find recursively all keys, sections or parameters
           matching a pattern

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
        results = []
        if text is not None:
            assert key is None, "text and key cannot both be provided!"
            assert section is None, "text and section cannot both be provided!"
            assert param is None, "text and param cannot both be provided!"
            key = text
            section = text
            param = text
        key = key if (key is None or case) else key.lower()
        section = section if (section is None or case) else section.lower()
        param = (
            param
            if (param is None or not isinstance(param, str) or case)
            else param.lower()
        )
        for item in self._find_patterns_generator(
            keypat=key, parampat=param, secpat=section, case=case
        ):
            results.append(item)
        return self.__class__._merge_PfsSections(results) if len(results) > 0 else None

    def _find_patterns_generator(
        self, keypat=None, parampat=None, secpat=None, keylist=[], case=False
    ):
        """Look for patterns in either keys, params or sections"""
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
                if keypat and keypat in kk:
                    yield from self._yield_deep_dict(keylist + [k], v)
                if self._param_match(parampat, v, case):
                    yield from self._yield_deep_dict(keylist + [k], v)

    @staticmethod
    def _yield_deep_dict(keys, val):
        """yield a deep nested dict with keys with a single deep value val"""
        for j in range(len(keys) - 1, -1, -1):
            d = {keys[j]: val}
            val = d
        yield d

    @staticmethod
    def _param_match(parampat, v, case):
        if parampat is None:
            return False
        if type(v) != type(parampat):
            return False
        if isinstance(v, str):
            vv = str(v) if case else str(v).lower()
            return parampat in vv
        else:
            return parampat == v

    def find_replace(self, old_value, new_value):
        """Update recursively all old_value with new_value"""
        for k, v in self.items():
            if isinstance(v, PfsSection):
                self[k].find_replace(old_value, new_value)
            elif self[k] == old_value:
                self[k] = new_value

    def copy(self) -> "PfsSection":
        """Return a copy of the PfsSection."""
        # is all this necessary???
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, PfsSection):
                d[key] = value.to_dict().copy()
        return self.__class__(d)

    def _to_txt_lines(self):
        lines = []
        self._write_with_func(lines.append, newline="")
        return lines

    def _write_with_func(self, func: Callable, level: int = 0, newline: str = "\n"):
        """Write pfs nested objects

        Parameters
        ----------
        func : Callable
            A function that performs the writing e.g. to a file
        level : int, optional
            Level of indentation (add 3 spaces for each), by default 0
        newline : str, optional
            newline string, by default "\n"
        """
        lvl_prefix = "   "
        for k, v in self.items():

            # check for empty sections
            if v is None:
                func(f"{lvl_prefix * level}[{k}]{newline}")
                func(f"{lvl_prefix * level}EndSect  // {k}{newline}{newline}")

            elif isinstance(v, List) and any(
                isinstance(subv, PfsSection) for subv in v
            ):
                # duplicate sections
                for subv in v:
                    if isinstance(subv, PfsSection):
                        subsec = PfsSection({k: subv})
                        subsec._write_with_func(func, level=level, newline=newline)
                    else:
                        subv = self._prepare_value_for_write(subv)
                        func(f"{lvl_prefix * level}{k} = {subv}{newline}")
            elif isinstance(v, PfsSection):
                func(f"{lvl_prefix * level}[{k}]{newline}")
                v._write_with_func(func, level=(level + 1), newline=newline)
                func(f"{lvl_prefix * level}EndSect  // {k}{newline}{newline}")
            elif isinstance(v, PfsNonUniqueList) or (
                isinstance(v, list) and all([isinstance(vv, list) for vv in v])
            ):
                if len(v) == 0:
                    # empty list -> keyword with no parameter
                    func(f"{lvl_prefix * level}{k} = {newline}")
                for subv in v:
                    subv = self._prepare_value_for_write(subv)
                    func(f"{lvl_prefix * level}{k} = {subv}{newline}")
            else:
                v = self._prepare_value_for_write(v)
                func(f"{lvl_prefix * level}{k} = {v}{newline}")

    def _prepare_value_for_write(self, v):
        """catch peculiarities of string formatted pfs data

        Parameters
        ----------
        v : str
            value from one pfs line

        Returns
        -------
            modified value
        """
        # some crude checks and corrections
        if isinstance(v, str):

            if len(v) > 5 and not ("PROJ" in v or "<CLOB:" in v):
                v = v.replace('"', "''")
                v = v.replace("\U0001F600", "'")

            if v == "":
                # add either '' or || as pre- and suffix to strings depending on path definition
                v = "''"
            elif v.count("|") == 2:
                v = f"{v}"
            else:
                v = f"'{v}'"

        elif isinstance(v, bool):
            v = str(v).lower()  # stick to MIKE lowercase bool notation

        elif isinstance(v, datetime):
            v = v.strftime("%Y, %m, %d, %H, %M, %S").replace(" 0", " ")

        elif isinstance(v, list):
            out = []
            for subv in v:
                out.append(str(self._prepare_value_for_write(subv)))
            v = ", ".join(out)

        return v

    def to_dict(self) -> dict:
        """Convert to (nested) dict (as a copy)"""
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, self.__class__):
                d[key] = value.to_dict()
        return d

    def to_dataframe(self, prefix: Optional[str] = None) -> pd.DataFrame:
        """Output enumerated subsections to a DataFrame

        Parameters
        ----------
        prefix : str, optional
            The prefix of the enumerated sections, e.g. "File\_",
            which can be supplied if it fails without this argument,
            by default None (will try to "guess" the prefix)

        Returns
        -------
        pd.DataFrame
            The enumerated subsections as a DataFrame

        Examples
        --------
        >>> pfs = mikeio.read_pfs("tests/testdata/pfs/lake.sw")
        >>> df = pfs.SW.OUTPUTS.to_dataframe(prefix="OUTPUT_")
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
            k = f"{prefix}{j+1}"
            res.append(self[k].to_dict())
        return pd.DataFrame(res, index=range(1, n_sections + 1))

    @classmethod
    def _merge_PfsSections(cls, sections: Sequence) -> "PfsSection":
        """Merge a list of PfsSections/dict"""
        assert len(sections) > 0
        a = sections[0]
        for b in sections[1:]:
            a = _merge_dict(a, b)
        return cls(a)
