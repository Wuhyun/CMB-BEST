# CMB-BEST
A code for CMB bispectrum estimation of primordial non-Gaussianity. For mathematical and implementational details, please refer to this paper.

## Dependencies

- Python3
- gcc
- Python libraries:
    - numpy
    - scipy
    - Cython
    - h5py
    - pandas
    - GetDist (optional for visualization)


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

A precomputed data file is required to use the library. Please download the HDF5 file below and keep in inside the "data" directory. At the moment, this data file is only given privately. Please contact wuhyun@kasi.re.kr if you are interested.


## Quick Start

Here is a simple Python script code to get started. Please refer to the Jupyter notebook under examples/ for further details.
```
import cmbbest as best

basis = best.Basis()
models = [best.Model("local")]
constraints = basis.constrain_models(models)

print(constraints.summary_df())
```