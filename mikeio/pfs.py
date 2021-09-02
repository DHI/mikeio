from datetime import datetime
import re
import yaml
import pandas as pd
import warnings

# from DHI.PFS import PFSFile, PFSSection, PFSParameter

from types import SimpleNamespace


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


class Pfs:
    def __init__(self, filename, encoding="cp1252"):

        warnings.warn(
            "Support for PFS files in mikeio is experimental. The API is likely to change!"
        )
        try:
            self._filename = filename
            self._pfs2yaml(encoding=encoding)
            self._data = yaml.load(self._yaml, Loader=yaml.CLoader)
            targets = list(self._data.keys())
            if len(targets) == 1:
                self._data = self._data[targets[0]]

            self.data = NestedNamespace(self._data)

            # create aliases
            if hasattr(self.data, "SPECTRAL_WAVE_MODULE"):
                self.data.SW = self.data.SPECTRAL_WAVE_MODULE
                self.data.SW.get_outputs = self._get_sw_outputs

            if hasattr(self.data, "HYDRODYNAMIC_MODULE"):
                self.data.HD = self.data.HYDRODYNAMIC_MODULE
                self.data.HD.get_outputs = self._get_hd_outputs

        except Exception:
            raise ValueError(
                f"""Support for PFS files in mikeio is limited
                and the file: {filename} could not be parsed."""
            )

    def _get_sw_outputs(self, included_only=False):
        return self.get_outputs("SPECTRAL_WAVE_MODULE", included_only=included_only)

    def _get_hd_outputs(self, included_only=False):
        return self.get_outputs("HYDRODYNAMIC_MODULE", included_only=included_only)

    def get_outputs(self, section, included_only=False):

        sub = self._data[section]["OUTPUTS"]
        n = sub["number_of_outputs"]

        sel_keys = [
            "file_name",
            "include",
            "type",
            "format",
            "first_time_step",
            "last_time_step",
            "use_end_time",
            "time_step_frequency",
        ]
        rows = []
        index = range(1, n + 1)
        for i in range(n):
            output = sub[f"OUTPUT_{i+1}"]
            row = {key: output[key] for key in sel_keys}

            rows.append(row)
        df = pd.DataFrame(rows, index=index)

        if included_only:
            df = df[df.include == 1]
        return df

    def _pfs2yaml(self, encoding=None):

        with (open(self._filename, encoding=encoding)) as f:
            pfsstring = f.read()

        lines = pfsstring.split("\n")

        output = []
        output.append("---")

        self._level = 0

        for line in lines:
            adj_line = self._parse_line(line)
            output.append(adj_line)

        self._yaml = "\n".join(output)

    def _parse_line(self, line):
        s = line.strip()
        s = re.sub(r"\s*//.*", "", s)

        if len(s) > 0:
            if s[0] == "[":
                s = s.replace("[", "")

            if s[-1] == "]":
                s = s.replace("]", ":")

        s = s.replace("//", "").replace("|", "")

        if len(s) > 0 and s[0] != "!":
            if "=" in s:
                idx = s.index("=")

                key = s[0:idx]
                key = key.strip()
                value = s[(idx + 1) :]
                key = key.lower()

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


class PfsCore:
    "PfsCore is no longer available in mikeio"

    def __init__(self, filename, target=None):
        raise NotImplementedError("PfsCore is no longer available in mikeio")
