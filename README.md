# pydhi
Facilitate creating, reading and writing dfs0, dfs2, and dfs3 files

## Install package from the dist directory:
    pip install py_dhi_dfs-0.0.1-py3-none-any.whl   

	% Assumes MIKE installed already on the computer. Add install directory to PYTHONPATH from windows command line:
	% set PYTHONPATH=%PYTHONPATH%;"C:\Program Files (x86)\DHI\2019\bin\x64"

# Examples

## Reading dfs0 file into pandas dataframe
	from pydhi import dfs0 as dfs0
	dfs = dfs0.dfs0()
	ts = dfs.read_to_pandas(dfs0file)

## Create non-equidistant dfs0
	dfs0file = r'C:\test\randomEQC.dfs0'
	data = np.random.random([1000, 2])
	data[2, :] = np.nan
	start_time = datetime.datetime(2017, 1, 1)
	time_vector = []
	for i in range(1000):
		time_vector.append( start_time + datetime.timedelta(hours=i*0.1) )
	title = 'Hello Test'
	names = ['VarFun01', 'NotFun']
	variable_type = [100000, 100000]
	unit = [1000, 1000]
	data_value_type = [0, 1]

	dfs = dfs0.dfs0()
	dfs.create_non_equidistant_calendar(dfs0file=dfs0file, data=data, time_vector=time_vector,
										names=names, title=title, variable_type=variable_type, unit=unit,
										data_value_type=data_value_type)

## Create equidistant dfs0											
	dfs0file = r'C:\test\randomEQC.dfs0'
	data = np.random.random([1000, 2])
	data[2, :] = np.nan
	start_time = datetime.datetime(2017, 1, 1)
	timeseries_unit = 1402
	title = 'Hello Test'
	names = ['VarFun01', 'NotFun']
	variable_type = [100000, 100000]
	unit = [1000, 1000]
	data_value_type = [0, 1]
	dt = 5
	dfs = dfs0.dfs0()
	dfs.create_equidistant_calendar(dfs0file=dfs0file, data=data, start_time=start_time,
									timeseries_unit=timeseries_unit, dt=dt, names=names,
									title=title, variable_type=variable_type, unit=unit,
									data_value_type=data_value_type)

## Read dfs2 data
	dfs2File = r"C:\test\random.dfs2"
	dfs = dfs2.dfs2()
	data = dfs.read(dfs2File, [0])[0]
	data = data[0]

## DFS Utilities to querry variable type, time series types (useful when creating a new dfs file)
	dfsUtil = dfs_util.dfs_util()
	dfsUtil.type_list()
	dfsUtil.timestep_list()
	
	
# Created by Marc-Etienne Ridler (mer@dhigroup.com)
python setup.py sdist bdist_wheel

