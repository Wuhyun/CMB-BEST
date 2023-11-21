DIR := ${CURDIR}
SRC := ${DIR}/src
OBJ := ${DIR}/objects

PYTHON := python3

cmbbest:
	${PYTHON} setup.py build_ext --inplace

install:
	${PYTHON} -m pip install -e .

clean:
	rm -rf build/*
	rm -rf lib/*
	rm -rf cmbbest*.so
	rm -rf *.egg-info


