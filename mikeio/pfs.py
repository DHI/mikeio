from datetime import datetime
import re
import yaml
import pandas as pd

import os # TO BE DELETED LATER? 


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


    # below added for pfs writing capabilities (developed by PIBS)

    def write_nested_dict(file_obj, dict_obj, indent): 
        # Iterate over all key-value pairs of dictionary
        for key, value in dict_obj.items():
            # If value is dict type, then print nested dict 
            if isinstance(value, dict):
                file_obj.write(' ' * indent + '['+key+']\n')
                print(' ' * indent, key, ':', '{')
                write_nested_dict(file_obj, value, indent + 3)
                file_obj.write(' ' * indent + 'EndSect  // '+key+'\n')
            else:
                file_obj.write(' ' * indent + '%s = %s\n' % (key, value))

    def dict_to_pfs_file(PFS_dict, outputdirectory, outputfilename):
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        outfilepath = outputdirectory + outputfilename
        with open(outfilepath, 'w') as f:
            write_nested_dict(f, PFS_dict, indent = 0)

    def PFS_to_dict(SourceFile):
        #Read source PFS file
        file1 = open(SourceFile, 'r')
        Lines = file1.readlines()
        for i in enumerate(Lines):
            Lines[i[0]]=  i[1].strip()
        
        nr_Levels = 6 # The .sw setup file has only 6 sub levels. OK to leave this hardcoded but it is not ideal for other pfs files
        LevelNames = [('Level' + str(i) + 'Name') for i in range(1 , nr_Levels + 1)] # list comprehension to create the names of the levels.
        columnNames = ['PFS','Type', 'LevelNo','Key', 'KeyName', 'Value', 'Comment'] + LevelNames

        df = pd.DataFrame(columns = columnNames)
        df['PFS'] = Lines
        #Identify type of line in the .sw setup file and populate dataframe
        for i in df.index:
            if df.loc[i, 'PFS'] == "":
                df.loc[i, 'Type'] = 'blank'
            if df.loc[i, 'PFS'].startswith('//'):
                df.loc[i, 'Type'] = 'comment'
            if df.loc[i, 'PFS'].startswith('['):
                df.loc[i, 'Type'] = 'StartSect'
            if df.loc[i, 'PFS'].startswith('EndSect'):
                df.loc[i, 'Type'] = 'EndSect'
            if "=" in df.loc[i, 'PFS']:
                df.loc[i, 'Type'] = 'key_value_pair'
        
        # Assign level of dict to dataframe column. At the same time assign the names of the levels to the dataframe.
        df['LevelNo'] = 0
        for i in df.index:
            if i == 0:
                continue
            elif df.loc[i, 'Type'] == 'comment':
                df.loc[i, 'LevelNo'] = df.loc[i-1, 'LevelNo']       # assign level the same as previous
                df.loc[i, LevelNames] = df.loc[i-1, LevelNames] 
            elif df.loc[i, 'Type'] == 'blank':
                df.loc[i, 'LevelNo'] = df.loc[i-1, 'LevelNo']       # assign level the same as previous
                df.loc[i, LevelNames] = df.loc[i-1, LevelNames]  
            elif df.loc[i, 'Type'] == 'key_value_pair':
                df.loc[i, 'LevelNo'] = df.loc[i-1, 'LevelNo']       # assign level the same as previous
                df.loc[i, LevelNames] = df.loc[i-1, LevelNames] 
            elif df.loc[i, 'Type'] == 'StartSect':
                df.loc[i, 'LevelNo'] = df.loc[i-1, 'LevelNo'] + 1       # Add +1 to the level when passing a StartSect
                df.loc[i, LevelNames] = df.loc[i-1, LevelNames] 
                df.loc[i, 'Level' + str(df.loc[i, 'LevelNo']) + 'Name'] = str(df.loc[i, 'PFS'][1:-1])
            elif df.loc[i, 'Type'] == 'EndSect':
                df.loc[i, LevelNames] = df.loc[i-1, LevelNames]  
                df.loc[i, 'Level' + str(df.loc[i-1, 'LevelNo']) + 'Name'] = float("NaN")
                df.loc[i, 'LevelNo'] = df.loc[i-1, 'LevelNo'] - 1

        # split key value pairs and assign to dataframe
        for i in df.index:
            if df.loc[i, 'Type'] == 'key_value_pair':
                df.loc[i, 'Key'] = df.loc[i, 'PFS'].split(' = ')[0]
                df.loc[i, 'Value'] = df.loc[i, 'PFS'].split(' = ')[1]
        
        # Find all comments and assign to dataframe. We dont use this again. Does not seem worth the effort to eventually write it back to the pfs file
        for i in df.index:
            if '//' in df.loc[i, 'PFS']:
                df.loc[i, 'Comment'] =  df.loc[i, 'PFS'].split('//')[1]
            
        # Add brackets
        for i in range(1,nr_Levels+1):
            if i == 1:
                df['Level1Name'] = ("['" + df['Level1Name'] + "']")            
            else:
                df['Level'+str(i)+'Name'] = ("['" + df['Level'+str(i)+'Name'] + "']").fillna('')
        df['Key'] = ("['" + df['Key'] + "']").fillna('')
        
        df['KeyName'] = df['Level1Name']+df['Level2Name']+df['Level3Name']+df['Level4Name']+df['Level5Name']+df['Level6Name']+df['Key']  # This hardcode to 6 levels command is not ideal, need to look for alternative

        #Start with Hardcoded empty dict, then populate
        PFS_dict = {}
        #Select all key value pairs for which the values are NaN. Then create empty dicts
        df_dropna = df[['KeyName', 'Value']].dropna(subset=['KeyName', 'Value'], how='all') # drop rows where both key and value is NaN
        df_keyvalue_dicts = df_dropna.dropna().drop_duplicates() # drop any rows with NaN, and remove duplicates. We are left with only the key value pairs at the deepest level of the sub-dicts
        df_dictofdicts = df_dropna['KeyName'][df['Value'].isnull()].drop_duplicates() # the inverse of df_keyvalue_dicts

        
        for i in enumerate(df_dictofdicts):  # We need to print the dicts containing dicts first, otherwise they will write over the keyvalue_dicts. This changes up the pfs file sequence, but tests show that it does not seem to affect how it is read by the Mike engine.
            exec('PFS_dict' + i[1] + '= {}') # This exec command is not ideal, need to look for alternative
            
        for i in enumerate(df_keyvalue_dicts['KeyName'].index):
            exec('PFS_dict' + df_keyvalue_dicts.loc[i[1],'KeyName'] + ' = r"' + str(df_keyvalue_dicts.loc[i[1],'Value']) +'"' ) # This exec command is not ideal, need to look for alternative
        return PFS_dict