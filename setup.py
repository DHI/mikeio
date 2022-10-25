import setuptools

with open("mikeio/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mikeio",
    version="1.2.1",
    install_requires=[
        "mikecore>=0.2.1",
        "numpy>=1.15.0",  # first version with numpy.quantile
        "pandas>1.3",
        "scipy>1.0",
        "PyYAML",
        "tqdm",
        "xarray",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black==22.3.0",
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
            "shapely",
            "pyproj",
            "xarray",
            "netcdf4",
            "matplotlib",
            "jupyterlab",
        ],
        "test": ["pytest", "matplotlib!=3.5.0", "xarray"],
        "notebooks": [
            "nbformat",
            "nbconvert",
            "jupyter",
            "xarray",
            "netcdf4",
            "rasterio",
            "geopandas",
            "scikit-learn",
        ],
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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)
