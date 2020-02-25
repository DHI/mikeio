import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mikeio",
    version="0.2.1",
    install_requires=["pythonnet", "numpy", "datetime", "pandas"],
    author="Henrik Andersson",
    author_email="jan@dhigroup.com",
    description="Create, write and read DHI files, e.g. dfs0, dfs2, dfs3, dfsu, res1d and mesh files.",
    platform="windows_x64",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/mikeio",
    download_url="https://github.com/DHI/mikeio/archive/0.2.1.zip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
)
