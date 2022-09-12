from datetime import datetime
import re
import yaml
import pandas as pd

import os  # TO BE DELETED LATER?


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

        try:
            self._filename = filename
            self._pfs2yaml(encoding=encoding)
            self._data = yaml.load(self._yaml, Loader=yaml.CLoader)
            targets = list(self._data.keys())
            if len(targets) == 1:
                self._data = self._data[targets[0]]

            self.data = NestedNamespace(self._data)

            # create aliases
            # if hasattr(self.data, "SPECTRAL_WAVE_MODULE"):
            #    self.data.SW = self.data.SPECTRAL_WAVE_MODULE
            #    self.data.SW.get_outputs = self._get_sw_outputs

            # if hasattr(self.data, "HYDRODYNAMIC_MODULE"):
            #    self.data.HD = self.data.HYDRODYNAMIC_MODULE
            #    self.data.HD.get_outputs = self._get_hd_outputs

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

        s = s.replace("//", "")

        # check for pipes in filenames
        if s.count("|") == 2:
            s = s[0:-1].replace("|", "'|") + "|'"

        if len(s) > 0 and s[0] != "!":
            if "=" in s:
                idx = s.index("=")

                key = s[0:idx]
                key = key.strip()
                value = s[(idx + 1) :]
                # key = key.lower()

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

    # added by clcr 12.Sept.2022
    def catch_str_exceptions(self, v):
        """catch peculiarities of string formatted pfs data

        Args:
            v (str): value from one pfs line

        Returns:
            v: modified value
        """
        # some crude checks and corrections
        if isinstance(v, str):
            # catch scientific notation
            try:
                v = float(v)
            except ValueError:
                # add either '' or || as pre- and suffix to strings depending on path definition
                if v == "":
                    v = "''"
                elif (
                    v[0:2] == ".."
                ):  # if it begins with . (hinting at relative path, use ||)
                    v = f"|{v}|"
                else:
                    v = f"'{v}'"

        if isinstance(v, datetime):
            v = v.strftime("%Y, %#m, %#d, %#H, 0, 0")  # pfs-datetime output

        if isinstance(v, list):
            v = str(v)[1:-1]  # strip [] from lists
        return v

    def write_nested_output(self, f, nested_data, lvl):
        """
        similar to pibs write_nested_dict but able to handle pfs nested objects directly
        Args:
            f (file object): file object (to write to)
            nested_data (mikeio.pfs.NestedNamespace): object holding (modified or non-modified data)
            lvl (int): level of indentation, add a tab \t for each
        """
        from types import SimpleNamespace

        lvl_prefix = "\t"
        for k, v in vars(nested_data).items():
            # check if values are again a namespace instance / new level
            if isinstance(v, SimpleNamespace):
                f.write(f"{lvl_prefix * lvl}[{k}]\n")
                self.write_nested_output(f, v, lvl + 1)
                f.write(f"{lvl_prefix * lvl}EndSect  // {k}\n\n")

            else:
                # print(f"{lvl_prefix * lvl}{k} = {v}\n") # JUST FOR TESTING, TO BE REMOVED

                v = self.catch_str_exceptions(v)

                # write output
                f.write(f"{lvl_prefix * lvl}{k} = {v}\n")

    def write(self, fname_out):
        """
        similar to pibs write_nested_dict but able to handle pfs nested objects directly
        Args:
            fname_out (path): path and filename to write output to
            lvl (int): level of indentation, add a tab \t for each
        """

        with open(fname_out, "w") as f:
            # HEADER (TO BE MODIFIED LATER)
            f.write(
                f"// Created     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                r"// DLL         : C:\Program Files (x86)\DHI\MIKE Zero\2021\bin\x64\pfs2004.dll"
            )
            f.write("\n")
            f.write(r"// Version     : 19.0.0.14309")
            f.write("\n\n")
            f.write("[FemEngineSW]\n")

            self.write_nested_output(f, self.data, 1)

            f.write("EndSect")
