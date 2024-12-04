# DataQualityGroup-MTSClean

## Description
This is the experiment code for the MTSClean: "Efficient Constraint-based Cleaning for Multi-Dimensional Time Series Data". 
MTSClean is a constraint-based multi-dimensional time series data cleaning algorithm. 
It divides the data into different windows, mines the row and column constraints, and then cleans the data based on the constraints.
In addition, it also includes some constraint mining algorithms for time series data, such as RFDiscover, DAFDiscover and TSDD.

## Repository structure
- `cleaning_algorithmes/`: the implementation of MTSClean and other cleaning method
- `constraint_mining_algorithms/`: the implementation of constraint mining algorithms
- `datasets/`: a sample of the datasets used for experiments
- `experiments/`: the main program for comparison

## Comparative experiments
We compare MTSClean with other data cleaning methods, including IMR, SCREEN, and Smooth. We have implemented all of these methods in `cleaning_algorithmes/`.

## Requirements
- python 3.6.8 or later
- numpy
- pandas
- scipy
- deap
- pykalman
- multiprocessing
- tkinter
- tqdm

## Usage

### Example
Once you have prepared your environment, it is already runnable with a simple example. You just need to run:
```shell
python ./experiments/comparison.py
```
then you can see the output of the example.

### Configures
You can change the cleaning settings in `/experiments/comparison.py`. For example, you can put the dataset you need to test into `datasets/`, and then modify the file reading path in `/experiments/comparison.py` to run the data you need. Note that you need to remove the timestamp from the data file and convert it to a csv file before putting it into `datasets/`.

You can also run specific cleaning algorithms instead of all algorithms by adjusting the contents of ```algorithmes```.