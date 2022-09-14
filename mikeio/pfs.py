from types import SimpleNamespace
from typing import List
from collections import Counter
from datetime import datetime
import re
import yaml
import pandas as pd


def read_pfs(filename, encoding="cp1252"):
    """Read a pfs file to a Pfs object for further analysis/manipulation

    Parameters
    ----------
    filename: str or Path
        File name including full path to the pfs file.
    encoding: str, optional
        How is the pfs file encoded? By default 'cp1252'

    Returns
    -------
    mikeio.Pfs
        Pfs object which can be used for inspection, manipulation and writing
    """
    return Pfs(filename, encoding=encoding)


class PfsSection(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            self.__set_key_value(key, value, copy=True)

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
        if isinstance(value, dict):
            d = value.copy() if copy else value
            self.__setattr__(key, PfsSection(d))  #
        elif isinstance(value, List) and isinstance(value[0], dict):
            # multiple Sections with same name
            sections = []
            for v in value:
                d = v.copy() if copy else v
                sections.append(PfsSection(d))
            self.__setattr__(key, sections)
        else:
            self.__setattr__(key, value)

    def pop(self, key, *args):
        # if key in self.keys():
        #     self.__delattr__(key)
        return self.__dict__.pop(key, *args)

    def get(self, key, *args):
        return self.__dict__.get(key, *args)

    def clear(self):
        return self.__dict__.clear()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    # TODO: better name
    def update_recursive(self, key, value):
        """Update recursively all matches of key with value"""
        for k, v in self.items():
            if isinstance(v, self.__class__):
                self[k].update_recursive(key, value)
            elif k == key:
                self[k] = value

    def find_replace(self, old_value, new_value):
        """Update recursively all old_value with new_value"""
        for k, v in self.items():
            if isinstance(v, self.__class__):
                self[k].find_replace(old_value, new_value)
            elif self[k] == old_value:
                self[k] = new_value

    def copy(self):
        # is all this necessary???
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, self.__class__):
                d[key] = value.to_dict().copy()
        return self.__class__(d)

    def to_Pfs(self, rootname: str = None):
        return Pfs(self, rootname=rootname)

    def to_file(self, filename, rootname: str) -> None:
        Pfs(self, rootname=rootname).write(filename)

    def to_dict(self):
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, self.__class__):
                d[key] = value.to_dict()
        return d

    def to_dataframe(self, prefix: str = None) -> pd.DataFrame:
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


def parse_yaml_preserving_duplicates(src):
    class PreserveDuplicatesLoader(yaml.loader.Loader):
        pass

    def map_constructor(loader, node, deep=False):
        keys = [loader.construct_object(node, deep=deep) for node, _ in node.value]
        vals = [loader.construct_object(node, deep=deep) for _, node in node.value]
        key_count = Counter(keys)
        data = {}
        for key, val in zip(keys, vals):
            if key_count[key] > 1:
                if key not in data:
                    data[key] = []
                data[key].append(val)
            else:
                data[key] = val
        return data

    PreserveDuplicatesLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, map_constructor
    )
    return yaml.load(src, PreserveDuplicatesLoader)


class Pfs:
    """Create a Pfs object for reading, writing and manipulating pfs files

    Parameters
    ----------
    input: dict, PfsSection, str or Path
        Either a file name (including full path) to the pfs file
        to be read or dictionary-like structure later to be written
        to a pfs file.
    encoding: str, optional
        How is the pfs file encoded? By default cp1252
    rootname: str, optional
        If the input is dictionary or PfsSection object the
        name of the root element can be specified
        If the input is a file the rootname is read from the file,
        by default None.
    """

    def __init__(self, input, encoding="cp1252", rootname=None):
        n_roots = 1
        if isinstance(input, PfsSection):
            root_section = input
        elif isinstance(input, dict):
            d = PfsSection(input)
            root_section = PfsSection(d)
        else:
            d = self._read_pfs_file_to_dict(input, encoding)
            rootname = list(d.keys())
            if len(rootname) == 1:
                rootname = rootname[0]
                root_section = PfsSection(d[rootname])
            else:
                n_roots = len(rootname)
                root_section = [PfsSection(d[k]) for k in rootname]

        self.data = root_section
        self._rootname = rootname
        if rootname is not None:
            if n_roots == 1:
                setattr(self, rootname, self.data)
            else:
                for n, section in zip(rootname, root_section):
                    setattr(self, n, section)
        self._add_all_FM_aliases()

    def _read_pfs_file_to_dict(self, filename, encoding):
        self._filename = filename
        try:
            yml = self._pfs2yaml(filename, encoding)
            # d = yaml.load(yml, Loader=yaml.CLoader)
            d = parse_yaml_preserving_duplicates(yml)
        except Exception:
            raise ValueError(
                f"""Support for PFS files in mikeio is limited
                and the file: {filename} could not be parsed."""
            )
        # root_keys = list(d.keys())
        # if len(root_keys) != 1:
        #    raise ValueError("Only pfs files with a single root element are supported")

        # _rootname = root_keys[0]
        return d  # , root_keys

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
        s = line.strip()
        s = re.sub(r"\s*//.*", "", s)  # remove comments

        if len(s) > 0:
            if s[0] == "[":
                s = s.replace("[", "")

            if s[-1] == "]":
                s = s.replace("]", ":")

        s = s.replace("//", "")

        # check for pipes in filenames
        if s.count("|") == 2:
            parts = s.split("|")
            s = parts[0] + "'|" + parts[1] + "|'" + parts[2]

        if len(s) > 0 and s[0] != "!":
            if "=" in s:
                idx = s.index("=")

                key = s[0:idx]
                key = key.strip()
                value = s[(idx + 1) :]

                if s.count("'") == 2:  # This is a quoted string and not a list
                    s = s
                else:
                    if "," in value:
                        value = f"[{value}]"

                if key == "start_time":
                    v = eval(value)
                    value = datetime(*v)

                s = f"{key}: {value}"

        if "EndSect" in line:
            s = ""

        ws = " " * 2 * self._level
        adj_line = ws + s

        s = line.strip()
        if len(s) > 0 and s[0] == "[":
            self._level += 1
        if "EndSect" in line:
            self._level -= 1

        return adj_line

    def _prepare_value_for_write(self, v):
        """catch peculiarities of string formatted pfs data

        Args:
            v (str): value from one pfs line

        Returns:
            v: modified value
        """
        # some crude checks and corrections
        if isinstance(v, str):
            try:
                # catch scientific notation
                v = float(v)
            except ValueError:
                # add either '' or || as pre- and suffix to strings depending on path definition
                if v == "":
                    v = "''"
                elif v.count("|") == 2:
                    v = f"{v}"
                else:
                    v = f"'{v}'"

        if isinstance(v, datetime):
            # v = v.strftime("%Y, %#m, %#d, %#H, %M, %S") # pfs-datetime output
            # v = v.strftime("%Y, %-m, %-d, %-H, %-M, %-S")  # no zero padding
            v = v.strftime("%Y, %m, %d, %H, %M, %S").replace(" 0", " ")

        if isinstance(v, list):
            v = str(v)[1:-1]  # strip [] from lists
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
            if isinstance(v, List) and isinstance(v[0], PfsSection):
                # duplicate sections
                for subv in v:
                    self._write_nested_PfsSections(f, PfsSection({k: subv}), lvl)
            elif isinstance(v, PfsSection):
                f.write(f"{lvl_prefix * lvl}[{k}]\n")
                self._write_nested_PfsSections(f, v, lvl + 1)
                f.write(f"{lvl_prefix * lvl}EndSect  // {k}\n\n")
            else:
                v = self._prepare_value_for_write(v)
                f.write(f"{lvl_prefix * lvl}{k} = {v}\n")

    def write(self, filename, rootname: str = None):
        """Write object to a pfs file

        Parameters
        ----------
        filename: str
            Full path and filename of pfs to be created.
        rootname: str, optional
            If the Pfs object was not created by reading an existing
            pfs file, then its root element may not have a name.
            It can be provided here. By default None.
        """
        from mikeio import __version__ as mikeio_version

        rootname = self._rootname if rootname is None else rootname
        if rootname is None:
            raise ValueError("Name of root element has not been provided")

        with open(filename, "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"// Created     : {now}\n")
            f.write(r"// By          : MIKE IO")
            f.write("\n")
            f.write(rf"// Version     : {mikeio_version}")
            f.write("\n\n")
            f.write(f"[{rootname}]\n")

            self._write_nested_PfsSections(f, self.data, 1)

            f.write(f"EndSect  // {rootname}\n")
