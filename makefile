DIR := ${CURDIR}
SRC := ${DIR}/src
OBJ := ${DIR}/objects


cmbbest:
	python setup.py build_ext --inplace

clean:
	rm -rf build/*
	rm -rf lib/*
	rm -rf cmbbest*.so
	rm -rf *.egg-info


