from __future__ import annotations
import re
import warnings
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from ._pfssection import PfsNonUniqueList, PfsSection


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

        self._ALIAS_LIST = {"_ALIAS_LIST"}  # ignore these in key list
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
            target_list = self._parse_pfs(filename, encoding, unique_keywords)
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

    def _parse_pfs(
        self,
        filename: str | Path | TextIO,
        encoding: str | None,
        unique_keywords: bool = False,
    ) -> list[dict[str, Any]]:
        """Parse PFS file directly to list of dictionaries."""
        if hasattr(filename, "read"):  # To read in memory strings StringIO
            pfsstring = filename.read()
        else:
            pfsstring = Path(filename).read_text(encoding=encoding)

        lines = pfsstring.splitlines()

        # Stack to track nested sections: [(section_name, section_dict), ...]
        stack: list[tuple[str, dict[str, Any]]] = []
        root_sections: list[dict[str, Any]] = []

        # Track multiline string values
        in_multiline_string = False
        multiline_key = ""
        multiline_value_parts: list[str] = []

        for line in lines:
            s = line.strip()
            s = self._strip_comments(s).strip()

            if not s or s.startswith("!"):
                continue

            # Handle multiline string continuation
            if in_multiline_string:
                multiline_value_parts.append(s)
                # Check if this line ends the multiline string
                if s.endswith("'"):
                    # End of multiline string - join with spaces
                    full_value = " ".join(multiline_value_parts)
                    value = self._parse_pfs_value(full_value)
                    if stack:
                        _, current_dict = stack[-1]
                        self._add_to_dict(
                            current_dict, multiline_key, value, unique_keywords
                        )
                    in_multiline_string = False
                    multiline_key = ""
                    multiline_value_parts = []
                continue

            # Check for section start
            if s.startswith("[") and s.endswith("]"):
                section_name = s[1:-1].strip()
                new_section: dict[str, Any] = {}
                stack.append((section_name, new_section))

            # Check for section end
            elif "EndSect" in s:
                if stack:
                    section_name, section_dict = stack.pop()

                    if not stack:
                        # This is a root section
                        root_sections.append({section_name: section_dict})
                    else:
                        # Add to parent section
                        _, parent_dict = stack[-1]
                        self._add_to_dict(
                            parent_dict, section_name, section_dict, unique_keywords
                        )

            # Key-value pair
            elif "=" in s:
                # Check for malformed brackets in key part only
                if "]]" in s or "[[" in s:
                    idx = s.index("=")
                    key_part = s[:idx]
                    if "]]" in key_part or "[[" in key_part:
                        raise ValueError(
                            f"Malformed PFS file: found ']]' or '[[' in line: {s}"
                        )

                idx = s.index("=")
                key = s[:idx].strip()
                value_str = s[idx + 1 :].strip()

                # Check if this starts a multiline string
                # A multiline string starts with ' but the ENTIRE value doesn't end with '
                # (not just checking last character, as that could be part of a list like '', '', "", "")
                if value_str.startswith("'"):
                    # Count quotes to see if they're balanced
                    # Simple heuristic: if there's only one ', it's likely multiline
                    single_quote_count = value_str.count("'")
                    if single_quote_count == 1:
                        in_multiline_string = True
                        multiline_key = key
                        multiline_value_parts = [value_str]
                        continue

                value = self._parse_pfs_value(value_str)

                if stack:
                    _, current_dict = stack[-1]
                    self._add_to_dict(current_dict, key, value, unique_keywords)

            # Check for malformed brackets (lines that don't have =)
            elif "]]" in s or "[[" in s:
                raise ValueError(f"Malformed PFS file: found ']]' or '[[' in line: {s}")

        return root_sections

    def _add_to_dict(
        self,
        target_dict: dict[str, Any],
        key: str,
        value: Any,
        unique_keywords: bool,
    ) -> None:
        """Add key-value pair to dictionary, handling duplicates."""
        if key in target_dict:
            # Handle duplicates
            existing = target_dict[key]

            if unique_keywords and not isinstance(value, dict):
                # Warn about non-section duplicates when unique_keywords is True
                warnings.warn(
                    f"Keyword {key} defined multiple times (first will be used). Value: {value}"
                )
                # Keep the first value
                return

            # Convert to PfsNonUniqueList if not already
            if not isinstance(existing, PfsNonUniqueList):
                target_dict[key] = PfsNonUniqueList([existing])

            target_dict[key].append(value)
        else:
            target_dict[key] = value

    def _strip_comments(self, s: str) -> str:
        """Remove comments from line while preserving quoted strings."""
        pattern = r"(\".*?\"|\'.*?\')|//.*"

        def replacer(match):  # type: ignore
            # Keep strings intact, remove comments
            return match.group(1) if match.group(1) else ""

        return re.sub(pattern, replacer, s)

    def _parse_pfs_value(self, value_str: str) -> Any:
        """Parse a value string into appropriate Python type."""
        if len(value_str) == 0:
            return []

        # Special case: pipe-delimited strings
        if (
            value_str.startswith("|")
            and value_str.endswith("|")
            and value_str.count("|") == 2
        ):
            return self._parse_token(value_str)

        # Special case: MULTIPOLYGON
        if "MULTIPOLYGON" in value_str:
            return value_str

        # Special case: values enclosed in double quotes should not be split
        # (even though they may contain commas)
        if value_str.startswith('"') and value_str.endswith('"'):
            # But if there's a comma, it's treated as a list per PFS convention
            if "," in value_str:
                tokens = self._split_line_by_comma(value_str)
                parsed_tokens = [
                    self._parse_token(t, context=value_str) for t in tokens
                ]
                return parsed_tokens
            else:
                return self._parse_token(value_str)

        # Check if it's a comma-separated list
        if "," in value_str:
            tokens = self._split_line_by_comma(value_str)
            parsed_tokens = [self._parse_token(t, context=value_str) for t in tokens]
            # Return as list if multiple values, single value otherwise
            return parsed_tokens if len(parsed_tokens) > 1 else parsed_tokens[0]
        else:
            return self._parse_token(value_str)

    def _split_line_by_comma(self, s: str) -> list[str]:
        """Split line by commas, respecting quoted strings."""
        # Manual parsing to respect quotes
        tokens: list[str] = []
        current_token: list[str] = []
        in_single_quote = False
        in_double_quote = False

        for i, char in enumerate(s):
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                current_token.append(char)
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                current_token.append(char)
            elif char == "," and not in_single_quote and not in_double_quote:
                tokens.append("".join(current_token))
                current_token = []
            else:
                current_token.append(char)

        # Add the last token
        if current_token:
            tokens.append("".join(current_token))

        return tokens

    def _parse_token(self, token: str, context: str = "") -> Any:
        """Parse a single token into appropriate Python type."""
        s = token.strip()

        # Handle pipe-delimited strings with special characters
        # Example: |file\path.dfs|
        if s.count("|") == 2 and "CLOB" not in context:
            prefix, content, suffix = s.split("|")
            if len(content) > 1 and content.count("'") > 0:
                # string containing single quotes that needs escaping
                warnings.warn(
                    f"The string {s} contains a single quote character which will be temporarily converted to \U0001f600 . If you write back to a pfs file again it will be converted back."
                )
                content = content.replace("'", "\U0001f600")
            # Return the pipe-delimited string as-is (will be stored as string)
            return f"|{content}|"

        # Replace double single quotes with regular quotes
        if len(s) > 2:
            s = s.replace("''", '"')

        # Remove surrounding quotes and parse the value
        if s.startswith("'") and s.endswith("'"):
            return s[1:-1]
        elif s.startswith('"') and s.endswith('"'):
            return s[1:-1]

        # Try to parse as number
        try:
            # Try integer first
            if "." not in s and "e" not in s.lower():
                return int(s)
            # Try float
            return float(s)
        except ValueError:
            pass

        # Try to parse as boolean
        if s.lower() == "true":
            return True
        elif s.lower() == "false":
            return False

        # Return as string
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
