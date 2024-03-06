# Generic

The generic module contains functionality that works for all types of dfs (dfs0, dfs1, dfs2, dfs3, dfsu) files: 

* [`concat()`](`mikeio.generic.concat`) - Concatenates files along the time axis
* [`extract()`](`mikeio.generic.extract`) - Extract timesteps and/or items to a new dfs file
* [`diff()`](`mikeio.generic.diff`) - Calculate difference between two dfs files with identical geometry
* [`sum()`](`mikeio.generic.sum`) - Calculate the sum of two dfs files
* [`scale()`](`mikeio.generic.scale`) - Apply scaling to any dfs file
* [`avg_time()`](`mikeio.generic.avg_time`) - Create a temporally averaged dfs file
* [`quantile()`](`mikeio.generic.quantile`) - Create a dfs file with temporal quantiles

## When to use the generic module

* The processing is not tied to the spatial dimension of the data
* When the files are large and you want to avoid reading the entire file into memory

## When not to use the generic module

* When you need processing depending on the spatial information in the file. For example, spatial interpolation, subsetting, etc.
* When you need more complex processing, not covered by the generic module
* When the input files data are not dfs files
* When the end result is not a dfs file

## Example

```python
>>> from mikeio import generic
>>> generic.concat(["fileA.dfs2", "fileB.dfs2"], "new_file.dfs2")
```

## More examples

See the [Generic notebook](../examples/Generic.qmd) for more examples.

