# N3L Example - IRIS

## Description
A common example of use neural network to classificate the inputs to a recognized pattern.
Data set and description reference: https://archive.ics.uci.edu/ml/datasets/Iris

It's designed with the idea that each output neuron give the probabily about each of the three possibily classification.

| Name | Case ID |
|------|---------|
| Iris-virginica | 0 |
| Iris-setosa | 1 |
| Iris-versicolor | 2 |

##### Results Graph
_MNS Range [0, 1]_

###### Activation Sigmoid - 1 Hidden Layer - 3 Hidden neurons per Layer - Learning Rate 0.05
![Results](http://i64.tinypic.com/m93mkx.png)
**MNS** Higher is better, **MNE** Lower is better

###### Activation Swish - 2 Hidden Layer - 3 Hidden neurons per Layer - Learning Rate 0.01
![Results](http://i65.tinypic.com/14xdm2o.png)
**MNS** Higher is better, **MNE** Lower is better

## Build
Execute the command below in this folder to compile the example:

```
gcc iris.c -o iris -ln3l
```

## Usage
The number of neurons and layers is fixed by code, anyway there are others parameters that you can edit at run time. For example the number of iterations, the learning rate or the bias terms.
Too see the complete list you have to run the following command:
```
./iris -h
```

Output:
```
IRIS Example - N3L v. 1.2.9
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>

Usage: ./iris [options]

Options:
	-b [n]         Set the bias term in the network. Default: 0
	-h             Show this help with the options list.
	-i [n]         Number of iterations. Default: 1
	-l             Enable learning with backpropagation.
	-m             No log at all. Note: Disable -v option.
	-o [filename]  After the number of iterations provided, save the neural network state.
                 Note: It works only if used with option -s.
	-p             Enable the progress viewer. Active -m, Disable -v.
	-r [filename]  Initialize the neural network reading the number of neurons, layers and weights from a previous state saved.
	-s             After the number of iterations provided, save the neural network state. Default filename: iris.n3l
	-v [n]         Enable N3 Library to log with specified verbosity.
	               Value: 0 - Critical, 1 - High, 2 - Medium, 3 - Low, 4 - Pedantic.
```

### Examples

Run 10k iterations in learning mode and save results in `iris.n3l`

```
./iris -l -i 10000 -s
```

Run 4 iterations in forward mode reading network from `iris.100k.n3l`

```
./iris -i 4 -r iris.100k.n3l
```

The previous state can be improved with more iterations using `-s` and `-r` options together:

```
./iris -l -i 10000 -r iris.100k.n3l -s -o iris.110k.n3l
```

## Network description
The network is built with the following parameters:
- 4 Input neuron + 0 bias
- 2 Hidden neuron + 0 bias
- 3 Output neuron
- Hidden neurons with Sigmoid as activation function
- Output neuron with Sigmoid as activation function
- Learning rate set to 0.05

## Results
With defaults settings at around 100k iterations it provide good results >96%.

```
./iris -i 100000 -l
```

### Memory usage
#### Forward mode
In a test with 100k iterations the memory stucks at 156KB.

#### Backward mode
In a test with 100k iterations the memory oscillates from 172KB to 188K.

### Processing time
Processing time is evaluated on the following processor:

```
vendor_id	: GenuineIntel
model name	: Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
cache size	: 3072 KB
siblings	: 4
cpu cores	: 2
```

#### Forward mode
Using _sys_ time as reference, the result is around **3627 iterations/second**.

```
time ./iris -i 100000 -m
IRIS Example - N3L v. 1.2.9
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>


real	0m34,246s
user	0m5,813s
sys	0m36,275s
```

#### Backward mode
Using _sys_ time as reference, the result is around **714 iterations/second**.

```
time ./iris -i 100000 -m -l
IRIS Example - N3L v. 1.2.9
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>


real	0m53,292s
user	0m12,149s
sys	1m11,442s
```
