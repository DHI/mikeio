from datetime import date, datetime, timedelta
import re
import yaml
import pandas as pd
from typing import Union
import warnings

from DHI.PFS import PFSFile, PFSSection, PFSParameter

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
    def __init__(self, filename):

        warnings.warn(
            "Support for PFS files in mikeio is experimental. The API is likely to change!"
        )
        print(
            "Support for PFS files in mikeio is experimental. The API is likely to change!"
        )

        try:
            self._filename = filename
            self._pfs2yaml()
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
        index = range(1, n+1)
        for i in range(n):
            output = sub[f"OUTPUT_{i+1}"]
            row = {key: output[key] for key in sel_keys}

            rows.append(row)
        df = pd.DataFrame(rows, index=index)

        if included_only:
            df = df[df.include == 1]
        return df

    def _pfs2yaml(self):

        with (open(self._filename)) as f:
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


# TODO come up with a better name
class PfsCore:
    def __init__(self, filename, target=None):

        warnings.warn(
            "Support for PFS files in mikeio is experimental. The API is likely to change!"
        )
        print(
            "Support for PFS files in mikeio is experimental. The API is likely to change!"
        )

        self._pfs = PFSFile(filename)

        if target is None:
            self._target = self._find_target(filename)
        else:
            self._target = target

    def section(self, name: str, index=0):
        pfssection = self._pfs.GetTarget(self._target, 1).GetSection(name, index + 1)
        return Section(pfssection)

    def write(self, filename):
        """
        Write PFS file

        Parameters
        ----------
        filename, str
        """

        self._pfs.Write(filename)

    @property
    def start_time(self) -> datetime:
        target = self._pfs.GetTarget(self._target, 1)
        section = target.GetSection("TIME", 1)
        start_time = section.GetKeyword("start_time", 1)

        vals = [start_time.GetParameter(i).ToInt() for i in range(1, 7)]
        return datetime(*vals)

    @start_time.setter
    def start_time(self, value: datetime):
        target = self._pfs.GetTarget(self._target, 1)
        section = target.GetSection("TIME", 1)
        start_time = section.GetKeyword("start_time", 1)
        start_time.GetParameter(1).ModifyIntParameter(value.year)
        start_time.GetParameter(2).ModifyIntParameter(value.month)
        start_time.GetParameter(3).ModifyIntParameter(value.day)
        start_time.GetParameter(4).ModifyIntParameter(value.hour)
        start_time.GetParameter(5).ModifyIntParameter(value.minute)
        start_time.GetParameter(6).ModifyIntParameter(value.second)

    @property
    def end_time(self) -> datetime:

        start_time = self.start_time

        nt = self.section("TIME")["number_of_time_steps"].value
        dt = self.section("TIME")["time_step_interval"].value

        return start_time + timedelta(seconds=nt * dt)

    @end_time.setter
    def end_time(self, value: datetime):

        print("FOO")
        start_time = self.start_time
        dt = self.section("TIME")["time_step_interval"].value

        nt = int((value - start_time).total_seconds() / dt)

        self.section("TIME")["number_of_time_steps"] = nt

    def _find_target(self, filename):

        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            if "//" in line:
                text, comment = line.split("//")
            else:
                text = line

            if "[" in text:
                startidx = text.index("[") + 1
                endidx = text.index("]")
                target = text[startidx:endidx]
                return target

        return None


class Parameter:
    def __init__(self, parameter: PFSParameter):

        self._parameter = parameter

    def __repr__(self):

        return f"<Parameter>{self.value}"

    @property
    def value(self):
        par = self._parameter

        if par.IsInt():
            return par.ToInt()
        elif par.IsDouble():
            return par.ToDouble()
        elif par.IsFilename():
            return par.ToFileName()
        elif par.IsClob():
            return par.ToClob()
        else:
            return par.ToString()

    def modify(self, value: Union[int, float, str]):

        par = self._parameter

        if par.IsInt():
            par.ModifyIntParameter(value)
        elif par.IsDouble():
            par.ModifyDoubleParameter(value)
        elif par.IsFilename():
            par.ModifyFileNameParameter(value)
        elif par.IsClob():
            return par.ModifyClobParameter(value)
        else:
            return par.ModifyStringParameter(value)


class Section:
    def __init__(self, section: PFSSection):

        self._section = section

    def section(self, name, index=0):
        pfssection = self._section.GetSection(name, index + 1)
        return Section(pfssection)

    def keyword(self, name, index=0) -> Parameter:
        parameter = self._section.GetKeyword(name, 1).GetParameter(1)
        return Parameter(parameter)

    def __getattr__(self, key):
        return self.keyword(key)

    def __setitem__(self, key, item):
        self.keyword(key).modify(item)

    def __getitem__(self, key):
        return self.keyword(key)

