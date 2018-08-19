# N3L Example - XOR

## Description
This is the most common example provided by various resources online when they explain how a neural network works.

##### Results Graph
![Results](http://i65.tinypic.com/fkqzih.png)
**MNS** Higher is better, **MNE** Lower is better

## Build
Execute the command below in this folder to compile the xor example:

```
gcc xor.c -o xor -ln3l -lm -lpthread
```

## Usage
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

### Examples

Run 10k iterations in learning mode with 0.5 bias and save results in `xor.n3l`

```
./xor -l -i 10000 -s
```

Run 4 iterations in forward mode reading network from `xor.5k.default.n3l`

```
./xor -i 4 -r xor.5k.default.n3l
```

The previous state can be improved with more iterations using `-s` and `-r` options together:

```
./xor -l -i 10000 -r xor.5k.default.n3l -s -o xor.15k.default.n3l
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
### Memory usage
#### Forward mode
XOR example use 300 byte per iteration, plus a base of 2,840 bytes to initialize the network.
```
valgrind --leak-check=full ./xor -i 1 -m
==30621== Memcheck, a memory error detector
==30621== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==30621== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==30621== Command: ./xor -i 1 -m
==30621==
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>

==30621==
==30621== HEAP SUMMARY:
==30621==     in use at exit: 0 bytes in 0 blocks
==30621==   total heap usage: 30 allocs, 30 frees, 3,176 bytes allocated
==30621==
==30621== All heap blocks were freed -- no leaks are possible
==30621==
==30621== For counts of detected and suppressed errors, rerun with: -v
==30621== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

In a test with 100k iterations the memory stucks at 96KB.
In the whole process the allocated memory is ~34MB.

```
valgrind --leak-check=full ./xor -i 100000 -m
==30651== Memcheck, a memory error detector
==30651== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==30651== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==30651== Command: ./xor -i 100000 -m
==30651==
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>

==30651==
==30651== HEAP SUMMARY:
==30651==     in use at exit: 0 bytes in 0 blocks
==30651==   total heap usage: 1,200,018 allocs, 1,200,018 frees, 33,602,840 bytes allocated
==30651==
==30651== All heap blocks were freed -- no leaks are possible
==30651==
==30651== For counts of detected and suppressed errors, rerun with: -v
==30651== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

#### Backward mode
The results per iteration in learning mode are similar to the forward mode.

```
valgrind --leak-check=full ./xor -i 1 -l
==25573== Memcheck, a memory error detector
==25573== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==25573== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==25573== Command: ./xor -i 1 -l
==25573==
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>

Simulation property:
Read from file: False ( xor.n3l )
  Save to file: False ( xor.n3l )
 Learning rate: 1.000000
     Verbosity: -1
    Iterations: 1

[XOR] -- Iteration 1 on 1 --
[XOR]         Input 0: 0
[XOR]         Input 1: 0
[XOR]          Output: 0.754758
[XOR]          Target: 0
[XOR] Overall success: 0.000%
==25573==
==25573== HEAP SUMMARY:
==25573==     in use at exit: 0 bytes in 0 blocks
==25573==   total heap usage: 32 allocs, 32 frees, 3,216 bytes allocated
==25573==
==25573== All heap blocks were freed -- no leaks are possible
==25573==
==25573== For counts of detected and suppressed errors, rerun with: -v
==25573== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

In a test with 100k iterations the memory stucks at 96KB.
In the whole process the allocated memory is ~37MB.

```
valgrind --leak-check=full ./xor -i 100000 -l -m
==25749== Memcheck, a memory error detector
==25749== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==25749== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==25749== Command: ./xor -i 100000 -l -m
==25749==
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>

==25749==
==25749== HEAP SUMMARY:
==25749==     in use at exit: 0 bytes in 0 blocks
==25749==   total heap usage: 1,400,018 allocs, 1,400,018 frees, 37,602,840 bytes allocated
==25749==
==25749== All heap blocks were freed -- no leaks are possible
==25749==
==25749== For counts of detected and suppressed errors, rerun with: -v
==25749== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

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
Using _sys_ time as reference, the result is around **4210 iterations/second**.

```
time ./xor -i 100000 -m
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>


real	0m18,414s
user	0m3,888s
sys	0m23,749s
```

#### Backward mode
Using _sys_ time as reference, the result is around **2680 iterations/second**.

```
time ./xor -i 100000 -m -l
XOR Example - N3L v. 1.2.8
(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>


real	0m31,290s
user	0m5,927s
sys	0m37,287s
```
