# This will import the global_warming_csv_to_dfs0 function from the csv2dfs0.py script
from csv2dfs0 import global_warming_csv_to_dfs0


for year in [1990, 2000, 2010]:

    global_warming_csv_to_dfs0(
        csvfilename="tests/testdata/co2-mm-mlo.csv",
        dfs0filename=f"mauna_loa_co2_{year}.dfs0",
        year=year,
    )
