# Cleaning pipeline for the QM9-OR' dataset

Source:  [10.5281/zenodo.13380412](https://doi.org/10.5281/zenodo.13380412) (QM9-OR)

## Dependencies

* `numpy`
* `pandas`
* `scipy`
* `ase`
* `tqdm`
* `vmol`

```
pip install numpy pandas scipy ase tqdm vmol
```

## Steps

### 1. Extract data
```
./1_extract.py
```
Writes geometries to `xyz/??????.xyz` and specific rotations to `data_raw.csv`.

### 2. Add rotation sign and absolute value

```
./2_add_sign.py
```
Modifies `data_raw.csv`.

### 3. Determine point groups
```
./3_get_point_group.py
```
Modifies `data_raw.csv`.

### 4. Exclude achiral structures
```
./4_remove_bad.py
```
Output:
```
chiral and at least one |α|>1.0: 104387
achiral and all |α|≤1.0: 12187
weird: chiral and all |α|≤1.0: 356
weird: achiral and at least one |α|>1.0: 4486
weird: chiral and at least one |α|>1.0 but at least one |α|=0: 18
*** 121416 total, 104369 (86%) kept, 17047 (14%) removed
```
Creates `data.csv`.


### 5. Exlude ill-defined values
```
./5_remove_outliers.py
```
Removes outliers in correlations between rotation at different wavelengths.

Output:
```
589 vs 633
k= 0.57 b=  -1.36       R^2= 0.489      outliers= 14    threshold=20σ
k= 0.75 b=  -0.55       R^2= 0.929      outliers= 35    threshold=20σ
k= 0.81 b=  -0.16       R^2= 0.995      outliers= 14    threshold=40σ
k= 0.82 b=  -0.07       R^2= 0.997      outliers=  0    threshold=60σ

589 vs 355
k= 5.05 b= -36.62       R^2= 0.001      outliers= 13    threshold=20σ
k= 4.51 b= -15.67       R^2= 0.028      outliers= 46    threshold=20σ
k= 4.24 b=  -8.43       R^2= 0.094      outliers= 72    threshold=20σ
k= 4.25 b=   4.89       R^2= 0.195      outliers= 92    threshold=20σ
k= 5.47 b=   3.87       R^2= 0.439      outliers= 66    threshold=20σ
k= 5.71 b=   7.36       R^2= 0.584      outliers= 39    threshold=20σ
k= 5.70 b=   6.51       R^2= 0.629      outliers= 27    threshold=20σ
k= 5.69 b=   5.98       R^2= 0.656      outliers= 20    threshold=20σ
k= 5.69 b=   5.71       R^2= 0.674      outliers=  8    threshold=20σ
k= 5.69 b=   5.94       R^2= 0.681      outliers=  4    threshold=20σ
k= 5.70 b=   5.72       R^2= 0.684      outliers=  1    threshold=20σ
k= 5.69 b=   5.84       R^2= 0.685      outliers=  0    threshold=20σ

589 vs 633
k= 0.83 b=  -0.02       R^2= 0.999      outliers=  0    threshold=100σ

*** 104369 total, 103916 (99.6%) kept, 453 (0.4%) removed
```
Modifies `data.csv`.

### 6. Exclude structures that become achiral without H atoms

```
./6_remove_achiral_noH.py
```
Output:
```
*** 103916 total, 102939 (99%) kept, 977 (1%) removed
```
Modifies `data.csv`.


### 7. Augment the dataset with the other enantiomers

```
./7_add_other.py`
```
Writes geometries to `xyz/x??????.xyz` and specific rotations and metadata to `data_both.csv`.


### Print out point group disctribution

```
./pointgroup_stats.py
```
path='data_raw.csv' col='point_group'
C1	104314
Cs	16063
C2v	452
C2	409
C2h	49
C3v	44
Ci	16
C3	12
D3h	12
C*v	6
D2d	6
D2h	6
D*h	5
Td	5
C3h	4
D3	4
D2	3
D3d	3
C4	1
D6h	1
S4	1

path='data.csv' col='point_group'
C1	102529
C2	398
C3	9
D3	2
D2	1

path='data.csv' col='point_group_noH'
C1	102515
C2	410
C3	11
D3	2
D2	1
```
