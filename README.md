# ECIF
Extended Crystallographic Information File, which allows you to put multiple crystal structures into the same .ecif file and add additional properties to facilitate various usage scenarios, such as machine learning data. Inspired by the SDF(structure data files) and RDkit PandasTools.

# ECIFPandasTools

This Python module provides some tools for handling the conversion between ECIF files and pandas dataframes. ECIF files are a file format used for storing formatted crystal structure information, while pandas dataframes are a data structure used for data analysis.


## Features

- `WriteEcif(df, out, idName='ID', cifColName='CIF', properties=None)`: Writes a pandas dataframe to an ECIF file. Each row in the dataframe is converted into an ECIF block, each block contains a CIF part and some additional properties.

- `LoadEcif(ecif_file, idName='ID', cifColName='CIF')`: Loads data from an ECIF file into a pandas dataframe. Each ECIF block is converted into a row in the dataframe.

- `CifBlock`: This is a class for handling ECIF blocks. It provides some methods for setting and getting properties, adding CIF lines, adding CIF from pymatgen structures, getting CIF, getting the entire block, getting pymatgen structures from CIF, adding the entire block, and writing to CIF files.

## Usage

First, you need to have a pandas dataframe that contains some pymatgen Structure objects. Then, you can use the `WriteEcif` function to write this dataframe to an ECIF file. For example:

```python
import pandas as pd
from pymatgen.core.structure import Structure
from ECIFPandasTools import WriteEcif

# Assume you have a dataframe named df, which contains a column of Structure objects named 'CIF'
WriteEcif(df, 'output.ecif', cifColName='CIF', properties=df.columns)
```

Then, you can use the `LoadEcif` function to load data from the ECIF file into a new dataframe. For example:

```python
from ECIFPandas

Tools

 import LoadEcif

df = LoadEcif('output.ecif', cifColName='CIF')
```

Note that both of these functions accept some optional parameters for specifying the names of certain columns in the dataframe, as well as additional properties to be included in the ECIF file.

Below is a snapshot of our data frame (df). It contains the fields ID, exfoliation energy (exfoliation_en) and crystal structure (CIF).

| ID | exfoliation_en | CIF |
| --- | --- | --- |
| mb-jdft2d-001 | 63.593833 | [[1.49323138 3.32688405 7.26257785] Hf, [3.326... |
| mb-jdft2d-002 | 134.86375 | [[1.85068084 4.37698238 6.93015769] As, [-1.63... |
| mb-jdft2d-003 | 43.114667 | [[-1.23770919e-16  2.02133251e+00  1.19727954e... |
| mb-jdft2d-004 | 240.715488 | [[2.39882726 2.39882726 2.53701553] In, [0.054... |
| mb-jdft2d-005 | 67.442833 | [[ -1.50082215  -0.86650009 -19.85028757] Nb, ... |
| ... | ... | ... |
| mb-jdft2d-632 | 26.426545 | [[ -2.38592122   1.37751225 -13.178104  ] Co, ... |
| mb-jdft2d-633 | 43.574286 | [[1.92920996 1.92920997 4.57868062] Ca, [1.929... |
| mb-jdft2d-634 | 88.808659 | [[4.53578337 0.         3.14900225] Pd, [ 9.07... |
| mb-jdft2d-635 | 132.26525 | [[4.41728901 2.2026463  1.81895292] Hg, [6.631... |
| mb-jdft2d-636 | 63.564333 | [[ 0.70613488 -1.21109143  1.03195663] Co, [ 2... |


To better understand the contents of the CIF field, we can look at the details of `df['CIF'][0]`. This is an example describing the position of the elements Hf, Si and Te in the crystal structure:

```python
Structure Summary
Lattice
    abc : 3.66730534 3.66730534 27.311209
 angles : 90.0 90.0 90.0
 volume : 367.31195815130786
      A : 3.66730534 0.0 2.245576873063498e-16
      B : -2.245576873063498e-16 3.66730534 2.245576873063498e-16
      C : 0.0 0.0 27.311209
    pbc : True True True
PeriodicSite: Hf0 (Hf) (1.493, 3.327, 7.263) [0.4072, 0.9072, 0.2659]
PeriodicSite: Hf1 (Hf) (3.327, 1.493, 3.049) [0.9072, 0.4072, 0.1116]
PeriodicSite: Si2 (Si) (3.327, 3.327, 5.156) [0.9072, 0.9072, 0.1888]
PeriodicSite: Si3 (Si) (1.493, 1.493, 5.156) [0.4072, 0.4072, 0.1888]
PeriodicSite: Te4 (Te) (3.327, 1.493, 8.659) [0.9072, 0.4072, 0.3171]
PeriodicSite: Te5 (Te) (1.493, 3.327, 1.652) [0.4072, 0.9072, 0.06049]
```