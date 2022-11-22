from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Sequence, Mapping
from collections import Counter
from datetime import datetime
import re
import warnings
import yaml
import pandas as pd


def read_pfs(filename, encoding="cp1252", unique_keywords=False):
    """Read a pfs file to a Pfs object for further analysis/manipulation

    Parameters
    ----------
    filename: str or Path
        File name including full path to the pfs file.
    encoding: str, optional
        How is the pfs file encoded? By default 'cp1252'
    unique_keywords: bool, optional
        Should the keywords in a section be unique? Some tools e.g. the
        MIKE Plot Composer allows non-unique keywords.
        If True: warnings will be issued if non-unique keywords
        are present and the first occurence will be used
        by default False

    Returns
    -------
    mikeio.Pfs
        Pfs object which can be used for inspection, manipulation and writing
    """
    return Pfs(filename, encoding=encoding, unique_keywords=unique_keywords)


class PfsNonUniqueList(list):
    pass


def merge_PfsSections(sections: Sequence[Mapping]):
    """Merge a list of PfsSections/dict"""
    assert len(sections) > 0
    a = sections[0]
    for b in sections[1:]:
        a = _merge_dict(a, b)
    return PfsSection(a)


def _merge_dict(a: Mapping, b: Mapping, path: Sequence = None):
    """merges dict b into dict a; handling non-unique keys"""
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dict(a[key], b[key], path + [str(key)])
            # elif a[key] == b[key]:
            #     pass  # same leaf value
            else:
                ab = list(a[key]) + list(b[key])
                a[key] = PfsNonUniqueList(ab)
        else:
            a[key] = b[key]
    return a


class PfsSection(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            self.__set_key_value(key, value, copy=True)

    def __repr__(self) -> str:
        # return json.dumps(self.to_dict(), indent=2)
        return yaml.dump(self.to_dict(), sort_keys=False)

    def __len__(self):
        return len(self.__dict__)

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
            if isinstance(v, self.__class__):
                self[k].update_recursive(key, value)
            elif k == key:
                self[k] = value

    def find_keys(self, pattern:str, case:bool=False):
        """Find recursively all keys matching a pattern"""
        results = []
        pattern = pattern if case else pattern.lower()
        for item in self._find_keys_generator(pattern, case=case):
            results.append(item)
        return merge_PfsSections(results) if len(results) > 0 else None

    def _find_keys_generator(self, pattern, keylist=[], case=False):
        for k, v in self.items():
            kk = str(k) if case else str(k).lower()
            if isinstance(v, self.__class__):
                yield from v._find_keys_generator(pattern, keylist + [k], case=case)
            elif pattern in kk:
                yield from self._yield_deep_dict(keylist + [k], v)

    @staticmethod
    def _yield_deep_dict(keys, val):
        """yield a deep nested dict with keys with a single deep value val"""
        for j in range(len(keys) - 1, -1, -1):
            d = {keys[j]: val}
            val = d
        yield d

    def find_params(self, pattern:str, case:bool=False):
        """Find recursively all parameters matching a pattern"""
        results = []
        pattern = pattern if case else pattern.lower()
        for item in self._find_params_generator(pattern, case=case):
            results.append(item)
        return merge_PfsSections(results) if len(results) > 0 else None

    def _find_params_generator(self, pattern, keylist=[], case=False):
        for k, v in self.items():
            vv = str(v) if case else str(v).lower()
            if isinstance(v, self.__class__):
                yield from v._find_params_generator(pattern, keylist + [k], case=case)
            elif pattern in vv:
                yield from self._yield_deep_dict(keylist + [k], v)

    def find_sections(self, pattern:str, case:bool=False):
        """Find recursively all sections matching a pattern"""
        results = []
        pattern = pattern if case else pattern.lower()
        for item in self._find_sections_generator(pattern, case=case):
            results.append(item)
        return merge_PfsSections(results) if len(results) > 0 else None

    def _find_sections_generator(self, pattern, keylist=[], case=False):
        for k, v in self.items():
            kk = str(k) if case else str(k).lower()
            if isinstance(v, self.__class__):
                if pattern in kk:
                    yield from self._yield_deep_dict(keylist + [k], v)
                else:
                    yield from v._find_sections_generator(pattern, keylist + [k], case=case)

    def search(self, key:str=None, *, section:str=None, param:str=None, case:bool=False):
        """Find recursively all keys, sections or parameters matching a pattern"""
        # NOTE: logically OR if mulitple conditions
        results = []
        key = key if (key is None or case) else key.lower()
        section = section if (section is None or case) else section.lower()
        param = param if (param is None or case) else param.lower()
        for item in self._find_patterns_generator(keypat=key, parampat=param, secpat=section, case=case):
            results.append(item)
        return merge_PfsSections(results) if len(results) > 0 else None
    
    def _find_patterns_generator(self, keypat=None, parampat=None, secpat=None, keylist=[], case=False):
        """Look for patterns in either keys, params or sections"""
        for k, v in self.items():
            kk = str(k) if case else str(k).lower()
            if parampat is not None:
                vv = str(v) if case else str(v).lower()
            if isinstance(v, self.__class__):
                if secpat and secpat in kk:
                    yield from self._yield_deep_dict(keylist + [k], v)
                else:
                    yield from v._find_patterns_generator(keypat, parampat, secpat, keylist=keylist + [k], case=case)
            elif keypat and keypat in kk:
                yield from self._yield_deep_dict(keylist + [k], v)   
            elif parampat and parampat in vv:
                yield from self._yield_deep_dict(keylist + [k], v)                    

    def find_replace(self, old_value, new_value):
        """Update recursively all old_value with new_value"""
        for k, v in self.items():
            if isinstance(v, self.__class__):
                self[k].find_replace(old_value, new_value)
            elif self[k] == old_value:
                self[k] = new_value

    def copy(self):
        """Return a copy of the PfsSection."""
        # is all this necessary???
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, self.__class__):
                d[key] = value.to_dict().copy()
        return self.__class__(d)

    def to_Pfs(self, name: str):
        """Convert to a Pfs object (with this PfsSection as the target)

        Parameters
        ----------
        name : str
            Name of the target (=key that refer to this PfsSection)

        Returns
        -------
        Pfs
            A Pfs object
        """
        return Pfs(self, names=[name])

    def to_file(self, filename, name: str) -> None:
        """Write to a Pfs file (providing a target name)"""
        Pfs(self, names=[name]).write(filename)

    def to_dict(self):
        """Convert to (nested) dict (as a copy)"""
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, self.__class__):
                d[key] = value.to_dict()
        return d

    def to_dataframe(self, prefix: str = None) -> pd.DataFrame:
        """Output enumerated subsections to a DataFrame

        Parameters
        ----------
        prefix : str, optional
            The prefix of the enumerated sections, e.g. "File_", by default None

        Returns
        -------
        pd.DataFrame
            The enumerated subsections as a DataFrame
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
    def from_dataframe(cls, df: pd.DataFrame, prefix: str) -> "PfsSection":
        """Create a PfsSection from a DataFrame"""
        d = {}
        for idx in df.index:
            key = prefix + str(idx)
            value = df.loc[idx].to_dict()
            d[key] = value
        return cls(d)


def parse_yaml_preserving_duplicates(src, unique_keywords=False):
    class PreserveDuplicatesLoader(yaml.loader.Loader):
        pass

    def map_constructor_duplicates(loader, node, deep=False):
        keys = [loader.construct_object(node, deep=deep) for node, _ in node.value]
        vals = [loader.construct_object(node, deep=deep) for _, node in node.value]
        key_count = Counter(keys)
        data = {}
        for key, val in zip(keys, vals):
            if key_count[key] > 1:
                if key not in data:
                    data[key] = PfsNonUniqueList()
                data[key].append(val)
            else:
                data[key] = val
        return data

    def map_constructor_duplicate_sections(loader, node, deep=False):
        keys = [loader.construct_object(node, deep=deep) for node, _ in node.value]
        vals = [loader.construct_object(node, deep=deep) for _, node in node.value]
        key_count = Counter(keys)
        data = {}
        for key, val in zip(keys, vals):
            if key_count[key] > 1:
                if isinstance(val, dict):
                    if key not in data:
                        data[key] = PfsNonUniqueList()
                    data[key].append(val)
                else:
                    warnings.warn(
                        f"Keyword {key} defined multiple times (first will be used). Value: {val}"
                    )
                    if key not in data:
                        data[key] = val
            else:
                data[key] = val
        return data

    constructor = (
        map_constructor_duplicate_sections
        if unique_keywords
        else map_constructor_duplicates
    )
    PreserveDuplicatesLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        constructor=constructor,
    )
    return yaml.load(src, PreserveDuplicatesLoader)


class Pfs:
    """Create a Pfs object for reading, writing and manipulating pfs files

    Parameters
    ----------
    input: dict, PfsSection, str or Path
        Either a file name (including full path) to the pfs file
        to be read or dictionary-like structure.
    encoding: str, optional
        How is the pfs file encoded? By default cp1252
    names: List[str], optional
        If the input is dictionary or PfsSection object the
        name of the root element (=target) MUST be specified.
        If the input is a file the names are instead read from the file,
        by default None.
    unique_keywords: bool, optional
        Should the keywords in a section be unique? Some tools e.g. the
        MIKE Plot Composer allows non-unique keywords.
        If True: warnings will be issued if non-unique keywords
        are present and the first occurence will be used
        by default False
    """

    def __init__(self, input, encoding="cp1252", names=None, unique_keywords=False):
        self._names = []
        self._targets = []

        if isinstance(input, (str, Path)) or hasattr(input, "read"):
            if names is not None:
                raise ValueError("names cannot be given as argument if input is a file")
            sections, names = self._read_pfs_file(input, encoding, unique_keywords)
        else:
            sections, names = self._parse_non_file_input(input, names)

        self._targets = sections
        self._names = names
        self._set_all_target_attr()

        self._add_all_FM_aliases()

    @property
    def data(self):
        return self._targets[0] if self.n_targets == 1 else self._targets

    @property
    def _targets_as_dict(self):
        target_key_count = Counter(self.names)
        d = dict()
        for n, target in zip(self.names, self._targets):
            if target_key_count[n] > 1:
                if n not in d:
                    d[n] = []
                d[n].append(target)
            else:
                d[n] = target
        return d

    @property
    def n_targets(self) -> int:
        """Number of targets (root sections)"""
        return len(self._targets)

    @property
    def names(self) -> List[str]:
        """Names of the targets (root sections) as a list"""
        return self._names

    @names.setter
    def names(self, new_names):
        new_names = [new_names] if isinstance(new_names, str) else new_names
        if len(new_names) != self.n_targets:
            raise ValueError(
                f"Number of target names must match number of targets ({self.n_targets})"
            )
        self._remove_all_target_attr()
        self._names = new_names
        self._set_all_target_attr()

    @property
    def is_unique(self) -> bool:
        """Are the target (root) names unique?"""
        return len(set(self.names)) == len(self.names)

    def add_target(self, section, name):
        if name is None:
            raise ValueError("name must be provided")
        section = PfsSection(section) if isinstance(section, dict) else section
        if not isinstance(section, PfsSection):
            raise ValueError("section wrong type; must be dict or PfsSection")
        self._targets.append(section)
        self._names.append(name)
        self._set_all_target_attr()

    def _remove_all_target_attr(self):
        """When renaming targets we need to remove all old target attr"""
        for n in set(self.names):
            delattr(self, n)

    def _set_all_target_attr(self):
        for n, section in self._targets_as_dict.items():
            setattr(self, n, section)

    def __repr__(self) -> str:
        out = ["<mikeio.Pfs>"]
        for n, sct in zip(self.names, self._targets):
            sct_str = str(sct).replace("\n", "")
            if len(sct_str) < 50:
                out.append(f"{n}: {sct_str}")
            else:
                out.append(f"{n}: {sct_str[:45]}...")
        return "\n".join(out)

    def to_dict(self):
        """Convert to nested dictionary"""
        d = dict()
        target_key_count = Counter(self.names)
        for n, target in zip(self.names, self._targets):
            if target_key_count[n] > 1:
                if n not in d:
                    d[n] = []
                d[n].append(target.to_dict())
            else:
                d[n] = target.to_dict()
        return d

    def _read_pfs_file(self, filename, encoding, unique_keywords=False):
        self._filename = filename
        try:
            yml = self._pfs2yaml(filename, encoding)
            target_list = parse_yaml_preserving_duplicates(yml, unique_keywords)
        except AttributeError:  # This is the error raised if parsing fails, try again with the normal loader
            target_list = yaml.load(yml, Loader=yaml.CFullLoader)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise ValueError(f"{filename} could not be parsed. " + str(e))
        sections = [PfsSection(list(d.values())[0]) for d in target_list]
        names = [list(d.keys())[0] for d in target_list]
        return sections, names

    @staticmethod
    def _parse_non_file_input(input, names):
        """dict/PfsSection or lists of these can be parsed"""
        if names is None:
            raise ValueError("'names' must be provided if input is not a file")
        if isinstance(names, str):
            names = [names]

        if isinstance(input, PfsSection):
            sections = [input]
        elif isinstance(input, dict):
            sections = [PfsSection(input)]
        elif isinstance(input, (List, Tuple)):
            if isinstance(input[0], PfsSection):
                sections = input
            elif isinstance(input[0], dict):
                sections = [PfsSection(d) for d in input]
            else:
                raise ValueError("List input must contain either dict or PfsSection")
        else:
            raise ValueError(
                f"Input of type ({type(input)}) could not be parsed (pfs file, dict, PfsSection, lists of dict or PfsSection)"
            )
        if len(names) != len(sections):
            raise ValueError(
                f"Length of names ({len(names)}) does not match length of target sections ({len(sections)})"
            )
        return sections, names

    def _add_all_FM_aliases(self) -> None:
        """create MIKE FM module aliases"""
        self._add_FM_alias("HD", "HYDRODYNAMIC_MODULE")
        self._add_FM_alias("SW", "SPECTRAL_WAVE_MODULE")
        self._add_FM_alias("TR", "TRANSPORT_MODULE")
        self._add_FM_alias("MT", "MUD_TRANSPORT_MODULE")
        self._add_FM_alias("EL", "ECOLAB_MODULE")
        self._add_FM_alias("ST", "SAND_TRANSPORT_MODULE")
        self._add_FM_alias("PT", "PARTICLE_TRACKING_MODULE")
        self._add_FM_alias("DA", "DATA_ASSIMILATION_MODULE")

    def _add_FM_alias(self, alias: str, module: str) -> None:
        """Add short-hand alias for MIKE FM module, e.g. SW, but only if active!"""
        if hasattr(self.data, module):
            mode_name = f"mode_of_{module.lower()}"
            mode_of = int(self.data.MODULE_SELECTION.get(mode_name, 0))
            if mode_of > 0:
                setattr(self, alias, self.data[module])

    def _pfs2yaml(self, filename, encoding=None) -> str:

        if hasattr(filename, "read"):  # To read in memory strings StringIO
            pfsstring = filename.read()
        else:
            with (open(filename, encoding=encoding)) as f:
                pfsstring = f.read()

        lines = pfsstring.split("\n")

        output = []
        output.append("---")

        self._level = 0

        for line in lines:
            adj_line = self._parse_line(line)
            output.append(adj_line)

        return "\n".join(output)

    def _parse_line(self, line: str) -> str:
        section_header = False
        s = line.strip()
        s = re.sub(r"\s*//.*", "", s)  # remove comments

        if len(s) > 0:
            if s[0] == "[":
                section_header = True
                s = s.replace("[", "")

                # This could be an option to create always create a list to handle multiple identical root elements
                if self._level == 0:
                    s = f"- {s}"

            if s[-1] == "]":
                s = s.replace("]", ":")

        s = s.replace("//", "")
        s = s.replace("\t", " ")

        if len(s) > 0 and s[0] != "!":
            if "=" in s:
                idx = s.index("=")

                key = s[0:idx]
                key = key.strip()
                value = s[(idx + 1) :].strip()

                if key == "start_time":
                    value = datetime.strptime(value, "%Y, %m, %d, %H, %M, %S").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                value = self._parse_param(value)
                s = f"{key}: {value}"

        if "EndSect" in line:
            s = ""

        ws = " " * 2 * self._level
        if self._level > 0:
            ws = "  " + ws  # TODO
        adj_line = ws + s

        if section_header:
            self._level += 1
        if "EndSect" in line:
            self._level -= 1

        return adj_line

    def _parse_param(self, value: str) -> str:
        if len(value) == 0:
            return "[]"

        if "," in value:
            tokens = self._split_line_by_comma(value)
            for j in range(len(tokens)):
                tokens[j] = self._parse_token(tokens[j])
            value = f"[{','.join(tokens)}]" if len(tokens) > 1 else tokens[0]
        else:
            value = self._parse_token(value)
        return value

    _COMMA_MATCHER = re.compile(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")

    def _split_line_by_comma(self, s: str):
        return self._COMMA_MATCHER.split(s)
        # import shlex
        # lexer = shlex.shlex(s)
        # lexer.whitespace += ","
        # lexer.quotes += "|"
        # lexer.wordchars += ",.-"
        # return list(lexer)

    def _parse_token(self, token: str) -> str:
        s = token.strip()

        if s.count("|") == 2:
            parts = s.split("|")
            if len(parts[1]) > 1 and parts[1].count("'") > 0:
                # string containing single quotes that needs escaping
                warnings.warn(
                    f"The string {s} contains a single quote character which will be temporarily converted to \U0001F600 . If you write back to a pfs file again it will be converted back."
                )
                parts[1] = parts[1].replace("'", "\U0001F600")
            s = parts[0] + "'|" + parts[1] + "|'" + parts[2]

        if len(s) > 2:  # ignore foo = ''
            s = s.replace("''", '"')

        return s

    def _prepare_value_for_write(self, v):
        """catch peculiarities of string formatted pfs data

        Args:
            v (str): value from one pfs line

        Returns:
            v: modified value
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

    def _write_nested_PfsSections(self, f, nested_data, lvl):
        """
        write pfs nested objects
        Args:
            f (file object): file object (to write to)
            nested_data (mikeio.pfs.PfsSection)
            lvl (int): level of indentation, add a tab \t for each
        """
        lvl_prefix = "   "
        for k, v in vars(nested_data).items():

            # check for empty sections
            NoneType = type(None)
            if isinstance(v, NoneType):
                f.write(f"{lvl_prefix * lvl}[{k}]\n")
                f.write(f"{lvl_prefix * lvl}EndSect  // {k}\n\n")

            elif isinstance(v, PfsSection):
                f.write(f"{lvl_prefix * lvl}[{k}]\n")
                self._write_nested_PfsSections(f, v, lvl + 1)
                f.write(f"{lvl_prefix * lvl}EndSect  // {k}\n\n")
            elif isinstance(v, PfsNonUniqueList):
                if len(v) == 0:
                    # empty list -> keyword with no parameter
                    f.write(f"{lvl_prefix * lvl}{k} = \n")
                for subv in v:
                    if isinstance(subv, PfsSection):
                        self._write_nested_PfsSections(f, PfsSection({k: subv}), lvl)
                    else:
                        subv = self._prepare_value_for_write(subv)
                        f.write(f"{lvl_prefix * lvl}{k} = {subv}\n")
            else:
                v = self._prepare_value_for_write(v)
                f.write(f"{lvl_prefix * lvl}{k} = {v}\n")

    def write(self, filename):
        """Write object to a pfs file

        Parameters
        ----------
        filename: str
            Full path and filename of pfs to be created.
        """
        from mikeio import __version__ as mikeio_version

        with open(filename, "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"// Created     : {now}\n")
            f.write(r"// By          : MIKE IO")
            f.write("\n")
            f.write(rf"// Version     : {mikeio_version}")
            f.write("\n\n")

            for name, target in zip(self.names, self._targets):
                f.write(f"[{name}]\n")
                self._write_nested_PfsSections(f, target, 1)
                f.write(f"EndSect  // {name}\n\n")
