# N3L Example - XOR

This is the most common example provided by various resources online when they explain how a neural network works.

The number of neurons and layers is fixed by code, anyway there are others parameters that you can edit at run time. For example the number of iterations, the learning rate or the bias terms.
Too see the complete list you have to run the following command:
```
./xor -h
```

Output:
```
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>

Usage: ./xor [options]

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
	-s             After the number of iterations provided, save the neural network state. Default filename: xor.n3l
	-v [n]         Enable N3 Library to log with specified verbosity.
	               Value: 0 - Critical, 1 - High, 2 - Medium, 3 - Low, 4 - Pedantic.
```

## Network description
The network is built with the following parameters:
- 2 Input neuron + 1 bias
- 3 Hidden neuron + 1 bias
- 1 Output neuron
- Hidden neurons with Sigmoid as activation function
- Output neuron with ReLu as activation function
- Learning rate set to 1

## Results
With defaults settings at around 5k iterations ( with bias set to 0.5 ) it starts to provide good results.

```
./xor -i 5000 -l -b 0.5
```

```
[XOR] -- Iteration 4997 on 5000 --
[XOR]         Input 0: 0
[XOR]         Input 1: 0
[XOR]          Output: 0.083858
[XOR]          Target: 0
[XOR] -- Iteration 4998 on 5000 --
[XOR]         Input 0: 1
[XOR]         Input 1: 0
[XOR]          Output: 0.929720
[XOR]          Target: 1
[XOR] -- Iteration 4999 on 5000 --
[XOR]         Input 0: 1
[XOR]         Input 1: 1
[XOR]          Output: 0.060637
[XOR]          Target: 0
[XOR] -- Iteration 5000 on 5000 --
[XOR]         Input 0: 0
[XOR]         Input 1: 1
[XOR]          Output: 0.930496
[XOR]          Target: 1
```
