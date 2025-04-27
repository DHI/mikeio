notebooks := $(wildcard *.py)
jupyters := $(notebooks:.py=.ipynb)

all: $(jupyters)

%.ipynb: %.py
	marimo export ipynb $< -o $@ --include-outputs
clean:
	rm -f *.ipynb
