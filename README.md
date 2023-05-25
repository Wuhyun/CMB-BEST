# CMB-BEST
A code for CMB bispectrum estimation of primordial non-Gaussianity. For mathematical and implementational details, please refer to [this paper](https://arxiv.org/abs/2305.14646). Please [contact me](mailto:wuhyun@kasi.re.kr) if you have any issues installing or using the code.

## Dependencies

- gcc or some other C compiler
- Python 3.7 or above
- Python packages:
    - Cython
    - numpy
    - scipy
    - h5py
    - pandas
    - GetDist (optional for visualization)

The packages can be installed using [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#) or [pip3](https://packaging.python.org/en/latest/tutorials/installing-packages/).

## Installation
First of all, please check if you have installed the dependencies listed above. If your default Python3 command is not 'python3', change it in the makefile accordingly.

Clone the repository by, e.g.,
```
git clone https://github.com/Wuhyun/CMB-BEST.git
```

Compile the library using
```
$ make
```
To install the library, run
```
$ make lib
```
This installs the library 'cmbbest' in development mode. To update the library, running
 ```
 git pull
 make
 ```
anytime will do the trick.


## Data

A precomputed data file is required to use the package. Please download the HDF5 file from this [Google Drive link](https://drive.google.com/file/d/10BGPCXAtkWF23eZi4R_P-PYVd4NBtGnq/view?usp=share_link) and keep the file inside the "data" directory.


## Quick Start

Here is a simple Python script code to get started. Please refer to the Jupyter notebook under examples/ for further details.
```
import cmbbest as best

basis = best.Basis()
models = [best.Model("local")]
constraints = basis.constrain_models(models)

print(constraints.summarize())
```
