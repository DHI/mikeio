# Pfs

A PFS file is a text file with a tree structure that contains parameters and settings for MIKE tools and engines. MIKE IO can read, modify and create PFS files. 

The content of the PFS file is similar to a nested dictionary. The root element is often called the *target*. Some PFS files have multiple root elements. The below sections are called *PFS Sections* which can be nested and contain key-value pairs called *keywords* and *parameters*. 

When a PFS file is read with MIKE IO a Pfs object is created. It will contain one or more PfsSection objects - one for each target. The PfsSections will typically contain other PfsSections together with a number of key-value pairs. The PfsSection object is similar to a dictionary. 


```python
>>> import mikeio
>>> pfs = mikeio.read_pfs("file.pfs")
>>> pfs
TODO
```

## Read




## Update

The PfsSection object can be modified. Existing values can be changes, new key-value pairs can be added, subsections can added or removed. 

### Modify existing keyword


### Add new key-value pair


### Add new section


## Write 

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
>>> pfs = mikeio.Pfs(d, names="MYTOOL")
>>> pfs.write("new.pfs")
```

Multiple targets can be achieved by providing list of dictionaries. 




## Pfs API

```{eval-rst}
.. autofunction:: mikeio.read_pfs
.. autoclass:: mikeio.Pfs
	:members:
	:inherited-members:
```

## PfsSection API

```{eval-rst}
.. autoclass:: mikeio.PfsSection
	:members:
	:inherited-members:
```
