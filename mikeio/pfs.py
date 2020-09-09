from datetime import datetime
import yaml


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

        if hasattr(self.data, "HYDRODYNAMIC_MODULE"):
            self.data.HD = self.data.HYDRODYNAMIC_MODULE

    def __getitem__(self, x):

        return self._data[x]

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

        if len(s) > 0:
            if s[0] == "[":
                s = s.replace("[", "")

            if s[-1] == "]":
                s = s.replace("]", ":")

        s = s.replace("//", "#").replace("|", "")  # TODO

        if len(s) > 0 and s[0] != "!":
            if "=" in s:
                idx = s.index("=")

                key = s[0:idx]
                key = key.strip()
                value = s[(idx + 1) :]
                key = key.lower()

                if s.count("'") == 2:  # This is a quoted string and not an array
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

