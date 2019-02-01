import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py-dhi-dfs",
    version="0.0.1",
	install_requires=['pythonnet','numpy', 'datetime','pandas'],
    author="Marc-Etienne Ridler",
    author_email="mer@dhigroup.com",
    description="A package that works with the DHI dfs libraries to facilitate creating, writing and reading dfs0, dfs2, and dfs3 files.",
	platform="windows_x64",
	license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/py-dhi-dfs.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Windows 10",
    ],
)