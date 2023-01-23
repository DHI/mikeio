# Pfs

A PFS file is a text file with a tree structure that contains parameters and settings for MIKE tools and engines. MIKE IO can read, modify and create PFS files. 

## The PFS file

The content of the PFS file is similar to a nested dictionary. The root element is often called the *target*. Some PFS files have multiple root elements. The below sections are called *PFS Sections* which can be nested and contain key-value pairs called *keywords* and *parameters*. 

```
[TARGET1]
   keywordA = parameterA
   [SECTION1]
      keywordB = parameterB
      keywordC = parameterC
      [SECTION2]
         keywordD = parameterD         
      EndSect  // SECTION2 
   EndSect  // SECTION1 
EndSect  // TARGET1

[TARGET2]
   keywordE = parameterE 
   [SECTION3]
      keywordF = parameterF
   EndSect  // SECTION3 
EndSect  // TARGET2
```

## Read

When a PFS file is read with MIKE IO, a `PfsDocument` object is created. It will contain one or more `PfsSection` objects - one for each target. The PfsSections will typically contain other PfsSections together with a number of key-value pairs. 

A PFS file is read using `mikeio.read_pfs()`:

```python
>>> import mikeio
>>> pfs = mikeio.read_pfs("concat.mzt")
```

### PfsDocument

The `PfsDocument` is the MIKE IO equivalent to a PFS file. Its targets can be accessed by their name (as properties), like this:  

```python
>>> pfs.txconc
CLSID: TxConc.dll
TypeName: txconc
CREATEDTIME: '2020-03-11T15:24:45'
(...)
```

Or by the `pfs.targets` object (which is a list of PfsSections). Each of the targets is a `PfsSection` object consisting of key-value pairs (keyword-parameter) and other PfsSections. 

The `PfsDocument` object is similar to a dictionary. You can loop over its contents with `items()`, `keys()` and `values()` like a dictionary. 


### PfsSection

The `PfsSection` object is also similar to a dictionary. You can loop over its contents with `items()`, `keys()` and `values()` like a dictionary. 

```python
>>> pfs.txconc.keys()
dict_keys(['CLSID', 'TypeName', 'CREATEDTIME', 'MODIFIEDTIME', 'NOTES', 'Setup'])
```

You can access a specific parameter with the `get()` method: 

```python
>>> pfs.txconc.get("CLSID")
'TxConc.dll'
```

Or as a property with dot-notation---which is prefered in most cases as it is more readable: 

```python
>>> pfs.txconc.CLSID
'TxConc.dll'
```

A PfsSection can be converted to a dictionary with the to_dict() method: 

```python
>>> pfs.txconc.Setup.File_1.to_dict()
{'InputFile': '|.\\tide1.dfs1|', 'Items': 1}
```

If a PfsSection contains enumerated subsections, they can be converted to a pandas DataFrame with the `to_dataframe()` method: 

```python
>>> pfs.txconc.Setup.to_dataframe(prefix="File_")
        InputFile  Items
1  |.\tide1.dfs1|      1
2  |.\tide2.dfs1|      1
```



### Unique or non-unique keywords

Depending on the engine intended for reading the PFS file it may or may not make sense to have multiple identical keywords in the same PfsSection. MIKE 21/3 and the marine tools does *not* support non-unique keywords---if non-unique keywords are present, only the first will be read and the presence is most likely a mistake made by hand-editing the file. In other tools, e.g. MIKE Plot Composer, non-unique keywords are used a lot. How MIKE IO shall deal with non-unique keywords can be specified using the `unique_keywords` argument in the `read_pfs()` method: 

```python
>>> import mikeio
>>> pfs = mikeio.read_pfs("myplot.plt", unique_keywords=False)
```

If a PfsSection contains non-unique PfsSections or keywords and `unique_keywords=False`, the repeated key will only appear once and the corresponding value will be a list. 


## Update

The PfsSection object can be modified. Existing values can be changed, new key-value pairs can be added, subsections can added or removed. 



### Modify existing keyword

It is very simple to modify an existing keyword: 

```python
>>> pfs.txconc.Setup.Name = "new name"
```


### Add new key-value pair

A new key-value pair can be added, like in a dictionary, in this way: 

```python
>>> pfs.txconc.Setup["NewKeyword"] = 12.0
```


### Add new section as a copy of another section

Often a PfsSection is added using an existing PfsSection as a template. 

```python
>>> s = pfs.txconc.Setup.File_1.copy()
>>> s.InputFile = '|.\tide3.dfs1|'
>>> pfs.txconc.Setup["File_3"] = s
```



### Add new section from a dictionary

A PfsSection can be created from a dictionary and then added to another PfsSection like any other key-value pair: 

```python
>>> d = {'InputFile': '|.\\tide4.dfs1|', 'Items': 1}
>>> s = mikeio.PfsSection(d)
>>> pfs.txconc.Setup["File_4"] = s
```



## Write to file

A Pfs object can be written to a PFS file using the `write` method. 

```python
>>> pfs.write("new.pfs")
```


## Create new Pfs files

A new PFS file can be created from dictionary in the following way: 

```python
>>> import mikeio
>>> d = dict(
        key1=1,
        lst=[0.3, 0.7],
        file_name=r"|path\file.dfs0|",
        start_time=datetime(2019, 7, 1, 0, 0, 0),        
    )
>>> pfs = mikeio.PfsDocument({"MYTOOL": d})
>>> pfs.write("new.pfs")
```

Multiple targets can be achieved by providing list of dictionaries. 




## PfsDocument API

```{eval-rst}
.. autofunction:: mikeio.read_pfs
.. autoclass:: mikeio.PfsDocument
	:members:
	:inherited-members:
```

## PfsSection API

```{eval-rst}
.. autoclass:: mikeio.PfsSection
	:members:
	:inherited-members:
```
