import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mikeio",
    version="0.3.0",
    install_requires=["pythonnet", "numpy", "pandas"],
    author="Henrik Andersson",
    author_email="jan@dhigroup.com",
    description="A package that works with the DHI dfs libraries to facilitate creating, writing and reading dfs, res1d and mesh files.",
    platform="windows_x64",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/mikeio",
    download_url="https://github.com/DHI/mikeio/archive/0.3.0.zip",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
    ],
)
