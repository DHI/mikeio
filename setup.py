import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mikeio",
    version="0.6.1_dev",
    install_requires=["pythonnet", "numpy", "pandas", "scipy", "pyyaml"],
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
    },
    author="Henrik Andersson",
    author_email="jan@dhigroup.com",
    description="A package that uses the DHI dfs libraries to create, write and read dfs, res1d and mesh files.",
    platform="windows_x64",
    license="MIT",
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
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
    ],
)
