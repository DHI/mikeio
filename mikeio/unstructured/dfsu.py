import os
import warnings
import numpy as np
from datetime import datetime, timedelta

from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import DfsFactory
from DHI.Generic.MikeZero.DFS.dfsu import DfsuFile, DfsuFileType, DfsuBuilder, DfsuUtil


from ..dutil import get_item_info, get_valid_items_and_timesteps
from ..dataset import Dataset
from ..dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
    asNumpyArray,
    to_dotnet_array,
    asnetarray_v2,
)
from ..eum import ItemInfo
from ..helpers import safe_length
from .unstructuredbase import (
    UnstructuredType,
    _UnstructuredGeometry,
    get_nodes_from_source,
    get_elements_from_source,
)
from .mesh import Mesh


class Dfsu(_UnstructuredGeometry):

    _filename = None
    _source = None
    _deletevalue = None

    _n_timesteps = None
    _start_time = None
    _timestep_in_seconds = None

    _n_items = None
    _items = None
    _dtype = np.float64

    def __init__(self, filename, dtype=np.float64):
        """
        Create a Dfsu object

        Parameters
        ---------
        filename: str
            dfsu or mesh filename
        dtype: np.dtype, optional
            default np.float64, valid options are np.float32, np.float64
        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        self._filename = filename
        if not os.path.isfile(filename):
            raise Exception(f"file {filename} does not exist!")

        _, ext = os.path.splitext(filename)

        if ext == ".mesh":
            msh = Mesh(filename)
            self._set_geometry_from_mesh(msh)
        else:
            self._read_header(filename)

        self._dtype = dtype

    def __repr__(self):
        out = []
        if self._type is not None:
            out.append(self.type_name)
        out.append(f"Number of elements: {self.n_elements}")
        out.append(f"Number of nodes: {self.n_nodes}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        if not self.is_2d:
            out.append(f"Number of sigma layers: {self.n_sigma_layers}")
        if (
            self._type == UnstructuredType.DfsuVerticalProfileSigmaZ
            or self._type == UnstructuredType.Dfsu3DSigmaZ
        ):
            out.append(f"Max number of z layers: {self.n_layers - self.n_sigma_layers}")
        if self._n_items is not None:
            if self._n_items < 10:
                out.append("Items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"Number of items: {self._n_items}")
        if self._n_timesteps is not None:
            if self._n_timesteps == 1:
                out.append(f"Time: time-invariant file (1 step) at {self._start_time}")
            else:
                out.append(
                    f"Time: {self._n_timesteps} steps with dt={self._timestep_in_seconds}s"
                )
                out.append(f"      {self._start_time} -- {self.end_time}")
        return str.join("\n", out)

    def _set_geometry_from_mesh(self, mesh):
        self._projstr = mesh.projection_string
        self._type = UnstructuredType.Mesh

        # geometry
        self._nc, self._codes, self._n_nodes, self._node_ids = get_nodes_from_source(
            mesh._source
        )

        (
            self._n_elements,
            self._element_table_dotnet,
            self._element_ids,
        ) = get_elements_from_source(mesh._source)

    def _read_header(self, filename):
        """
        Read header of dfsu file and set object properties
        """
        dfs = DfsuFile.Open(filename)
        self._source = dfs
        self._projstr = dfs.Projection.WKTString
        self._type = UnstructuredType(dfs.DfsuFileType)
        self._deletevalue = dfs.DeleteValueFloat

        # geometry
        self._set_nodes_from_source(dfs)
        self._set_elements_from_source(dfs)

        if not self.is_2d:
            self._n_layers = dfs.NumberOfLayers
            self._n_sigma = dfs.NumberOfSigmaLayers

        # items
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))

        # time
        self._start_time = from_dotnet_datetime(dfs.StartDateTime)
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        dfs.Close()

    def _set_nodes_from_source(self, source):
        self._nc, self._codes, self._n_nodes, self._node_ids = get_nodes_from_source(
            source
        )

    def _set_elements_from_source(self, source):
        (
            self._n_elements,
            self._element_table_dotnet,
            self._element_ids,
        ) = get_elements_from_source(source)

    @property
    def element_coordinates(self):
        # faster way of getting element coordinates than base class implementation
        if self._ec is None:
            self._ec = self._get_element_coords_from_source()
        return self._ec

    def _get_element_coords_from_source(self):
        xc = np.zeros(self.n_elements)
        yc = np.zeros(self.n_elements)
        zc = np.zeros(self.n_elements)
        _, xc2, yc2, zc2 = DfsuUtil.CalculateElementCenterCoordinates(
            self._source, to_dotnet_array(xc), to_dotnet_array(yc), to_dotnet_array(zc),
        )
        ec = np.column_stack([asNumpyArray(xc2), asNumpyArray(yc2), asNumpyArray(zc2)])
        return ec

    @property
    def deletevalue(self):
        """File delete value
        """
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items
        """
        return self._n_items

    @property
    def items(self):
        """List of items
        """
        return self._items

    @property
    def start_time(self):
        """File start time
        """
        return self._start_time

    @property
    def n_timesteps(self):
        """Number of time steps
        """
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds
        """
        return self._timestep_in_seconds

    # @timestep.setter
    # def timestep(self, value):
    #     if value <= 0:
    #         print(f'timestep must be positive scalar!')
    #     else:
    #         self._timestep_in_seconds = value

    @property
    def end_time(self):
        """File end time
        """
        return self.start_time + timedelta(
            seconds=((self.n_timesteps - 1) * self.timestep)
        )

    def read(self, items=None, time_steps=None, elements=None):
        """
        Read data from a dfsu file

        Parameters
        ---------
        filename: str
            dfsu filename
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: int or list[int], optional
            Read only selected time_steps
        elements: list[int], optional
            Read only selected element ids   

        Returns
        -------
        Dataset
            A dataset with data dimensions [t,elements]

        Examples
        --------
        >>> dfsu.read()
        <mikeio.DataSet>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dfsu.read(time_steps="1985-08-06 12:00,1985-08-07 00:00")
        <mikeio.DataSet>
        Dimensions: (5, 884)
        Time: 1985-08-06 12:00:00 - 1985-08-06 22:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        """

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)
        # time may have changes since we read the header
        # (if engine is continuously writing to this file)
        self._n_timesteps = dfs.NumberOfTimeSteps
        # TODO: add more checks that this is actually still the same file
        # (could have been replaced in the meantime)

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)
        # n_items = self.n_items #safe_length(dfs.ItemInfo)

        nt = self.n_timesteps  # .NumberOfTimeSteps

        items, item_numbers, time_steps = get_valid_items_and_timesteps(
            self, items, time_steps
        )

        n_items = len(item_numbers)

        if elements is None:
            n_elems = self.n_elements
            n_nodes = self.n_nodes
        else:
            node_ids, _ = self._get_nodes_and_table_for_elements(elements)
            n_elems = len(elements)
            n_nodes = len(node_ids)

        deletevalue = self.deletevalue

        data_list = []

        item0_is_node_based = False
        for item in range(n_items):
            # Initialize an empty data block
            if item == 0 and items[item].name == "Z coordinate":
                item0_is_node_based = True
                data = np.ndarray(shape=(len(time_steps), n_nodes), dtype=self._dtype)
            else:
                data = np.ndarray(shape=(len(time_steps), n_elems), dtype=self._dtype)
            data_list.append(data)

        t_seconds = np.zeros(len(time_steps), dtype=float)

        for i in range(len(time_steps)):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                src = itemdata.Data

                d = to_numpy(src)

                d[d == deletevalue] = np.nan

                if elements is not None:
                    if item == 0 and item0_is_node_based:
                        d = d[node_ids]
                    else:
                        d = d[elements]

                data_list[item][i, :] = d

            t_seconds[i] = itemdata.Time

        time = [self.start_time + timedelta(seconds=tsec) for tsec in t_seconds]

        dfs.Close()
        return Dataset(data_list, time, items)

    def write_header(
        self, filename, start_time=None, dt=None, items=None, elements=None, title=None,
    ):
        """Write the header of a new dfsu file

            Parameters
            -----------
            filename: str
                full path to the new dfsu file
            start_time: datetime, optional
                start datetime, default is datetime.now()
            dt: float, optional
                The time step (in seconds)
            items: list[ItemInfo], optional
            elements: list[int], optional
                write only these element ids to file
            title: str
                title of the dfsu file. Default is blank.

            Examples
            --------
            >>> msh = Mesh("foo.mesh")
            >>> n_elements = msh.n_elements
            >>> dfs = Dfsu(meshfilename)
            >>> nt = 1000
            >>> n_items = 10
            >>> items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]
            >>> with dfs.write_header(outfilename, items=items) as f:
            >>>     for i in range(1, nt):
            >>>         data = []
            >>>         for i in range(n_items):
            >>>             d = np.random.random((1, n_elements))
            >>>             data.append(d)
            >>>             f.append(data)
            """

        return self.write(
            filename=filename,
            data=[],
            start_time=start_time,
            dt=dt,
            items=items,
            elements=elements,
            title=title,
            keep_open=True,
        )

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
        keep_open=False,
    ):
        """Write a new dfsu file

        Parameters
        -----------
        filename: str
            full path to the new dfsu file
        data: list[np.array] or Dataset
            list of matrices, one for each item. Matrix dimension: time, x
        start_time: datetime, optional
            start datetime, default is datetime.now()
        dt: float, optional
            The time step (in seconds)
        items: list[ItemInfo], optional
        elements: list[int], optional
            write only these element ids to file
        title: str
            title of the dfsu file. Default is blank.
        keep_open: bool, optional
            Keep file open for appending
        """

        if isinstance(data, Dataset):
            items = data.items
            start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                if not data.is_equidistant:
                    raise Exception(
                        "Data is not equidistant in time. Dfsu requires equidistant temporal axis!"
                    )
                dt = (data.time[1] - data.time[0]).total_seconds()
            data = data.data

        n_items = len(data)
        n_time_steps = 0
        if n_items > 0:
            n_time_steps = np.shape(data[0])[0]

        if dt is None:
            if self.timestep is None:
                dt = 1
            else:
                dt = self.timestep  # 1 # Arbitrary if there is only a single timestep

        if start_time is None:
            if self.start_time is None:
                start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {start_time} as start time."
                )
            else:
                start_time = self.start_time
                warnings.warn(
                    f"No start time supplied. Using start time from source: {start_time} as start time."
                )

        if items is None:
            if n_items == 0:
                raise ValueError(
                    "Number of items unknown. Add (..., items=[ItemInfo(...)]"
                )
            items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]

        if title is None:
            title = ""

        file_start_time = to_dotnet_datetime(start_time)

        # spatial subset
        if elements is None:
            geometry = self
        else:
            geometry = self.elements_to_geometry(elements)
            if (not self.is_2d) and (geometry._type == UnstructuredType.Dfsu2D):
                # redo extraction as 2d:
                print("will redo extraction in 2d!")
                geometry = self.elements_to_geometry(elements, node_layers="bottom")
                if items[0].name == "Z coordinate":
                    # get rid of z-item
                    items = items[1 : (n_items + 1)]
                    n_items = n_items - 1
                    new_data = []
                    for j in range(n_items):
                        new_data.append(data[j + 1])
                    data = new_data

        # Default filetype;
        if geometry._type == UnstructuredType.Mesh:
            # create dfs2d from mesh
            dfsu_filetype = DfsuFileType.Dfsu2D
        else:
            # TODO: if subset is slice...
            dfsu_filetype = geometry._type.value

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            if items[0].name != "Z coordinate":
                raise Exception("First item must be z coordinates of the nodes!")

        xn = geometry.node_coordinates[:, 0]
        yn = geometry.node_coordinates[:, 1]

        # zn have to be Single precision??
        zn = to_dotnet_float_array(geometry.node_coordinates[:, 2])

        elem_table = []
        for j in range(geometry.n_elements):
            elem_nodes = geometry.element_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]
            elem_table.append(elem_nodes)
        elem_table = asnetarray_v2(elem_table)

        builder = DfsuBuilder.Create(dfsu_filetype)

        builder.SetNodes(xn, yn, zn, geometry.codes)
        builder.SetElements(elem_table)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)

        factory = DfsFactory()
        proj = factory.CreateProjection(geometry.projection_string)
        builder.SetProjection(proj)
        builder.SetTimeInfo(file_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

        for item in items:
            if item.name != "Z coordinate":
                builder.AddDynamicItem(
                    item.name, eumQuantity.Create(item.type, item.unit)
                )

        try:
            self._dfs = builder.CreateFile(filename)
        except IOError:
            print("cannot create dfsu file: ", filename)

        deletevalue = self._dfs.DeleteValueFloat

        try:
            # Add data for all item-timesteps, copying from source
            for i in range(n_time_steps):
                for item in range(n_items):
                    d = data[item][i, :]
                    d[np.isnan(d)] = deletevalue
                    darray = to_dotnet_float_array(d)
                    self._dfs.WriteItemTimeStepNext(0, darray)
            if not keep_open:
                self._dfs.Close()
            else:
                return self

        except Exception as e:
            print(e)
            self._dfs.Close()
            os.remove(filename)

    def append(self, data):
        """Append to a dfsu file opened with `write(...,keep_open=True)`

        Parameters
        -----------
        data: list[np.array]
        """

        deletevalue = self._dfs.DeleteValueFloat
        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]
        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                darray = to_dotnet_float_array(d)
                self._dfs.WriteItemTimeStepNext(0, darray)

    def close(self):
        "Finalize write for a dfsu file opened with `write(...,keep_open=True)`"
        self._dfs.Close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._dfs.Close()

    def to_mesh(self, outfilename):
        """write object to mesh file

        Parameters
        ----------
        outfilename : str
            path to file to be written
        """
        if self.is_2d:
            _ = self.element_table  # make sure element table has been constructured
            geometry = self
        else:
            geometry = self.to_2d_geometry()
            # TODO: print warning if sigma-z
        Mesh._geometry_to_mesh(outfilename, geometry)
