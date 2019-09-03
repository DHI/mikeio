from unittest import TestCase

import numpy as np
import pandas as pd
import datetime

from pydhi import dfs2 as dfs2
from pydhi import dfs_util as dfs_util


class test_dfs2(TestCase):

    def test_create_single_item(self):

        start_time = datetime.datetime(2012, 1, 1)

        # timeseries_unit = second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
        timeseries_unit = 1402
        dt = 12

        # See what variable are available (and filter with water level)
        util = dfs_util.dfs_util()
        #variable_types = util.type_list(search='water level')

        # from result we see Water Level is 100000
        variable_type = [100000]

        #possible_units = util.unit_list(variable_type, search='meter')
        # from result, we see meter is 1000
        unit = [1000]

        dfs2File = r"C:\test\random.dfs2"

        data = []
        d = np.random.random([100, 100, 30])
        d[10, :, :] = np.nan
        d[11, :, :] = 0
        d[12, :, :] = 1e-10
        d[13, :, :] = 1e10

        data.append(d)

        coordinate = ['UTM-33', 12.4387, 55.2257, 327]
        x0 = 0
        y0 = 0
        length_x = 100
        length_y = 100

        names = ['testing water level']
        title = 'test dfs2'

        dfs = dfs2.dfs2()

        # print(help(dfs.create_equidistant_calendar))

        dfs.create_equidistant_calendar(dfs2file=dfs2File, data=data, start_time=start_time,
                                        timeseries_unit=timeseries_unit, dt=dt, variable_type=variable_type,
                                        unit=unit, coordinate=coordinate, x0=x0, y0=y0, length_x=length_x,
                                        length_y=length_y, names=names, title=title)

    def test_create_multiple_item(self):

        start_time = datetime.datetime(2012, 1, 1)

        # timeseries_unit = second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
        timeseries_unit = 1402
        dt = 12

        # See what variable are available (and filter with water level)
        util = dfs_util.dfs_util()
        #variable_types = util.type_list(search='water level')

        # from result we see Water Level is 100000, Rainfall is 100004, drain time constant 100362
        variable_type = [100000, 100004, 100362]

        #possible_units = util.unit_list(variable_type, search='meter')
        # from result, we see meter is 1000 and milimeter is 1002, per second is 2605
        unit = [1000, 1002, 2605]

        dfs2File = r"C:\test\multiple.dfs2"

        data = []
        d = np.zeros([100, 100, 30]) + 1.0
        data.append(d)
        d = np.zeros([100, 100, 30]) + 2.0
        data.append(d)
        d = np.zeros([100, 100, 30]) + 3.0
        data.append(d)

        coordinate = ['UTM-33', 12.4387, 55.2257, 327]
        x0 = 0
        y0 = 0
        length_x = 200
        length_y = 200

        names = ['testing water level', 'testing rainfall', 'testing drain time constant']
        title = 'test dfs2'

        dfs = dfs2.dfs2()

        # print(help(dfs.create_equidistant_calendar))

        dfs.create_equidistant_calendar(dfs2file=dfs2File, data=data, start_time=start_time,
                                        timeseries_unit=timeseries_unit, dt=dt, variable_type=variable_type,
                                        unit=unit, coordinate=coordinate, x0=x0, y0=y0, length_x=length_x,
                                        length_y=length_y, names=names, title=title)

    def test_read(self):

        dfs2File = r"C:\test\random.dfs2"
        dfs = dfs2.dfs2()

        data = dfs.read(dfs2File, [0])[0]
        data = data[0]
        self.assertEqual(data[11, 0, 0], 0)
        self.assertEqual(np.isnan(data[10, 0, 0]), True)
        #self.assertEqual(data[12, 0, 0],  1e-10)

    def test_write(self):
        #dfs2File = r"C:\test\random.dfs2"
        #dfs = dfs2.dfs2()
        #d = np.zeros([100, 100, 30]) + 1.111
        #dfs.write(dfs2file=dfs2File, data=[d])
        print('Deleted Test')
