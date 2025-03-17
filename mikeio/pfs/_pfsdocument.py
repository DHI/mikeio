from __future__ import annotations
import re
import warnings
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import yaml

from ._pfssection import PfsNonUniqueList, PfsSection


def parse_yaml_preserving_duplicates(
    src: Any, unique_keywords: bool = False
) -> dict[str, Any]:
    class PreserveDuplicatesLoader(yaml.loader.Loader):
        pass

    def map_constructor_duplicates(loader: Any, node: Any, deep: bool = False) -> Any:
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

    def map_constructor_duplicate_sections(
        loader: Any, node: Any, deep: bool = False
    ) -> Any:
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


class PfsDocument(PfsSection):
    """Create a PfsDocument object for reading, writing and manipulating pfs files.

    Parameters
    ----------
    data: dict, PfsSection, str or Path
        Either a file name (including full path) to the pfs file
        to be read or dictionary-like structure.
    encoding: str, optional
        How is the pfs file encoded? By default cp1252
    unique_keywords: bool, optional
        Should the keywords in a section be unique? Some tools e.g. the
        MIKE Plot Composer allows non-unique keywords.
        If True: warnings will be issued if non-unique keywords
        are present and the first occurence will be used
        by default False

    """

    def __init__(
        self,
        data: TextIO
        | Mapping[str | PfsSection, Any]
        | Sequence[PfsSection]
        | str
        | Path,
        *,
        encoding: str = "cp1252",
        unique_keywords: bool = False,
    ) -> None:
        if isinstance(data, (str, Path)) or hasattr(data, "read"):
            names, sections = self._read_pfs_file(data, encoding, unique_keywords)  # type: ignore
        else:
            names, sections = self._parse_non_file_input(data)

        d = self._to_nonunique_key_dict(names, sections)
        super().__init__(d)

        self._ALIAS_LIST = set(["_ALIAS_LIST"])  # ignore these in key list
        if self._is_FM_engine:
            self._add_all_FM_aliases()

    @staticmethod
    def from_text(text: str) -> PfsDocument:
        """Create a PfsDocument from a string."""
        from io import StringIO

        f = StringIO(text)
        return PfsDocument(f)

    @staticmethod
    def _to_nonunique_key_dict(keys: Any, vals: Any) -> dict[Any, Any]:
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

    def keys(self) -> list[str]:  # type: ignore
        """Return a list of the PfsDocument's keys (target names)."""
        return [k for k, _ in self.items()]

    def values(self) -> list[PfsSection | PfsNonUniqueList]:  # type: ignore
        """Return a list of the PfsDocument's values (targets)."""
        return [v for _, v in self.items()]

    def items(self) -> list[tuple[str, PfsSection | PfsNonUniqueList]]:  # type: ignore
        """Return a new view of the PfsDocument's items ((key, value) pairs)."""
        return [(k, v) for k, v in self.__dict__.items() if k not in self._ALIAS_LIST]

    def to_dict(self) -> dict:
        """Convert to (nested) dict (as a copy)."""
        d = super().to_dict()
        _ = d.pop("_ALIAS_LIST")
        return d

    @staticmethod
    def _unravel_items(items: Callable) -> tuple[list, list]:
        rkeys = []
        rvals = []
        for k, v in items():
            if isinstance(v, PfsNonUniqueList):
                for subval in v:
                    rkeys.append(k)
                    rvals.append(subval)
            else:
                rkeys.append(k)
                rvals.append(v)
        return rkeys, rvals

    @property
    def targets(self) -> list[PfsSection]:
        """List of targets (root sections)."""
        _, rvals = self._unravel_items(self.items)
        return rvals

    @property
    def n_targets(self) -> int:
        """Number of targets (root sections)."""
        return len(self.targets)

    @property
    def is_unique(self) -> bool:
        """Are the target (root) names unique?"""
        return len(self.keys()) == len(self.names)

    @property
    def names(self) -> list[str]:
        """Names of the targets (root sections) as a list."""
        rkeys, _ = self._unravel_items(self.items)
        return rkeys

    def copy(self) -> PfsDocument:
        """Return a deep copy of the PfsDocument."""
        text = repr(self)

        return PfsDocument.from_text(text)

    def _read_pfs_file(
        self,
        filename: str | Path | TextIO,
        encoding: str | None,
        unique_keywords: bool = False,
    ) -> tuple[list[str], list[PfsSection]]:
        try:
            yml = self._pfs2yaml(filename, encoding)
            target_list = parse_yaml_preserving_duplicates(yml, unique_keywords)
        except AttributeError:  # This is the error raised if parsing fails, try again with the normal loader
            target_list = yaml.load(yml, Loader=yaml.CFullLoader)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise ValueError(f"{filename} could not be parsed. " + str(e))
        return PfsDocument._extract_names_from_list(target_list)  # type: ignore

    @staticmethod
    def _extract_names_from_list(
        targets: Sequence[PfsSection],
    ) -> tuple[list[str], list[PfsSection]]:
        names, sections = zip(
            *((k, PfsSection(v)) for target in targets for k, v in target.items())
        )
        return list(names), list(sections)

    @staticmethod
    def _parse_non_file_input(
        input: Mapping[str | PfsSection, Any] | Sequence[PfsSection],
    ) -> tuple[list[str], list[PfsSection]]:
        if isinstance(input, Sequence):
            return PfsDocument._extract_names_from_list(input)

        assert isinstance(input, Mapping), "input must be a mapping"
        names, sections = PfsDocument._unravel_items(input.items)
        for sec in sections:
            if not isinstance(sec, Mapping):
                raise ValueError(
                    "all targets must be PfsSections/dict (no key-value pairs allowed in the root)"
                )
        return names, sections

    @property
    def _is_FM_engine(self) -> bool:
        return "FemEngine" in self.names[0]

    def _add_all_FM_aliases(self) -> None:
        """create MIKE FM module aliases."""
        ALIASES = {
            "HD": "HYDRODYNAMIC_MODULE",
            "SW": "SPECTRAL_WAVE_MODULE",
            "TR": "TRANSPORT_MODULE",
            "MT": "MUD_TRANSPORT_MODULE",
            "EL": "ECOLAB_MODULE",
            "ST": "SAND_TRANSPORT_MODULE",
            "PT": "PARTICLE_TRACKING_MODULE",
            "DA": "DATA_ASSIMILATION_MODULE",
        }
        for alias, module in ALIASES.items():
            self._add_FM_alias(alias, module)

    def _add_FM_alias(self, alias: str, module: str) -> None:
        """Add short-hand alias for MIKE FM module, e.g. SW, but only if active!"""
        target = self.targets[0]
        if hasattr(target, module) and hasattr(target, "MODULE_SELECTION"):
            mode_name = f"mode_of_{module.lower()}"
            mode_of = int(target.MODULE_SELECTION.get(mode_name, 0))
            if mode_of > 0:
                setattr(self, alias, target[module])
                self._ALIAS_LIST.add(alias)

    def _pfs2yaml(
        self, filename: str | Path | TextIO, encoding: str | None = None
    ) -> str:
        if hasattr(filename, "read"):  # To read in memory strings StringIO
            pfsstring = filename.read()
        else:
            pfsstring = Path(filename).read_text(encoding=encoding)

        lines = pfsstring.splitlines()

        output = []
        output.append("---")

        _level = 0

        for line in lines:
            adj_line, _level = self._parse_line(line, _level)
            output.append(adj_line)

        return "\n".join(output)

    def _strip_comments(self, s: str) -> str:
        pattern = r"(\".*?\"|\'.*?\')|//.*"

        def replacer(match):  # type: ignore
            # Keep strings intact, remove comments
            return match.group(1) if match.group(1) else ""

        return re.sub(pattern, replacer, s)

    def _parse_line(self, line: str, level: int = 0) -> tuple[str, int]:
        section_header = False
        s = line.strip()
        s = self._strip_comments(s).strip()

        if len(s) > 0:
            if s[0] == "[":
                section_header = True
                s = s.replace("[", "")

                # This could be an option to create always create a list to handle multiple identical root elements
                if level == 0:
                    s = f"- {s}"

            if s[-1] == "]":
                s = s.replace("]", ":")

        # s = s.replace("//", "")
        s = s.replace("\t", " ")

        if len(s) > 0 and s[0] != "!":
            if "=" in s:
                idx = s.index("=")

                key = s[0:idx]
                key = key.strip()
                value = s[(idx + 1) :].strip()
                value = self._parse_param(value)
                s = f"{key}: {value}"

        if "EndSect" in line:
            s = ""

        ws = " " * 2 * level
        if level > 0:
            ws = "  " + ws  # TODO
        adj_line = ws + s

        if section_header:
            level += 1
        if "EndSect" in line:
            level -= 1

        return adj_line, level

    def _parse_param(self, value: str) -> str:
        if len(value) == 0:
            return "[]"
        if value[0] == "|" and value[-1] == "|":
            if value.count("|") == 2:
                return self._parse_token(value)
        if "MULTIPOLYGON" in value:
            return value
        if "," in value:
            tokens = self._split_line_by_comma(value)
            for j in range(len(tokens)):
                tokens[j] = self._parse_token(tokens[j], context=value)
            value = f"[{','.join(tokens)}]" if len(tokens) > 1 else tokens[0]
        else:
            value = self._parse_token(value)
        return value

    _COMMA_MATCHER = re.compile(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")

    def _split_line_by_comma(self, s: str) -> list[str]:
        return self._COMMA_MATCHER.split(s)

    def _parse_token(self, token: str, context: str = "") -> str:
        s = token.strip()

        # Example of complicated string:
        # '<CLOB:22,1,1,false,1,0,"",0,"",0,"",0,"",0,"",0,"",0,"",0,"",||,false>'
        if s.count("|") == 2 and "CLOB" not in context:
            prefix, content, suffix = s.split("|")
            if len(content) > 1 and content.count("'") > 0:
                # string containing single quotes that needs escaping
                warnings.warn(
                    f"The string {s} contains a single quote character which will be temporarily converted to \U0001f600 . If you write back to a pfs file again it will be converted back."
                )
                content = content.replace("'", "\U0001f600")
            s = f"{prefix}'|{content}|'{suffix}"

        if len(s) > 2:  # ignore foo = ''
            s = s.replace("''", '"')

        return s

    def write(self, filename: str | Path) -> None:
        """Write object to a pfs file.

        Parameters
        ----------
        filename: str, optional
            Full path and filename of pfs to be created.

        Notes
        -----
        To return the content as a string, use repr()

        """
        from mikeio import __version__ as mikeio_version

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""// Created     : {now}
// By          : MIKE IO
// Version     : {mikeio_version}

"""
        txt = header + "\n".join(self._to_txt_lines())

        Path(filename).write_text(txt)
