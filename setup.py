import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mikeio",
    version="0.7.dev1",
    install_requires=[
        "mikecore",
        "numpy",
        "pandas",
        "scipy",
        "pyyaml",
        "tqdm",
        "pyproj",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "sphinx",
            "sphinx-rtd-theme",
            "shapely",
            "xarray",
            "netcdf4",
            "matplotlib",
            "jupyterlab",
        ],
        "test": ["pytest", "matplotlib", "shapely"],
        "notebooks": ["ipykernel", "nbformat", "nbconvert", "jupyter", "xarray", "netcdf4", "rasterio", "geopandas"],
    },
    author="Henrik Andersson",
    author_email="jan@dhigroup.com",
    description="A package that uses the DHI dfs libraries to create, write and read dfs and mesh files.",
    license="BSD-3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/mikeio",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
