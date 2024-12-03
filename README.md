# DataQualityGroup-MTSClean

## Description
This is the experiment code for the MTSClean: "Efficient Constraint-based Cleaning for Multi-Dimensional Time Series Data". 
MTSClean is a constraint-based multi-dimensional time series data cleaning algorithm. 
It divides the data into different windows, mines the row and column constraints, and then cleans the data based on the constraints.
In addition, it also includes some constraint mining algorithms for time series data, such as RFDiscover, DAFDiscover and TSDD

## Requirements
- numpy
- pandas
- scipy
- deap
- pykalman
- multiprocessing
- tkinter
- tqdm

## Running
1. Remove the timestamp from the data file and convert it to a csv file.
2. run
```shell
python ./experiments/comparison.py
```