![Build](https://img.shields.io/badge/build-passing-green.svg) ![Status](https://img.shields.io/badge/status-alpha-yellow.svg) ![Version](https://img.shields.io/badge/version-1.2.9-lightgray.svg)
# N3 Library
A tiny C library for building neural network with the capability to define custom activation functions, learning rate, bias and others parameters.

Both forward and backpropagation are built to parallelize through threads the operations.

In forward propagation each layer execute the _activation functions_ in concurrent threads, when one layer end its own job, the results to the each next layer's neuron are collected with others concurrent threads. The whole process is executed from each layer from the input one to the output one.

In backward propagation first there is the delta evaluation from the output layer to the input layer, after the weight updating process from the input to the output. Each return to the previous layer and delta evaluation on neurons in the same layer, is built, even in this case, with concurrent threads.

## Build
The library is designed to run under GNU\Linux operating systems.

You can compile it with the following commands:
```
$ make
# make install
```

To clean the main folder you can run `make clean`.
To uninstall `make uninstall` as root user.

**Flags:**
* _debug=true_ add the arguments `-g -Ddebug_enable -pg`
* _extra=true_ add the arguments `-Wall -Wextra -ansi -pedantic`
* _flags=arguments_ add the arguments specified
* _destdir=mydir_ install the library in the specified path `mydir`

### Use it in projects
While compiling your projects you should include the `n3lib.h` header and compile with the option `-ln3l`. Example:

```c
#include <n3l/n3lib.h>

/* gcc myprog.c -ln3l */
```

## Quick Start

Build a network with N3 Library without many options it's fast, first you need to initialize the parameters and for this case there is the `N3LArgs` structure.
If you don't want to spend much time thinking at every parameter there is the function `n3l_get_default_args()` which set generically each one. After that you have only to specify the structure of the neural network with this parameters:
* `in_size`: number of neurons in the input layer
* `h_size`: number of neurons in the hidden layer
* `h_layers`: number of hidden layers
* `out_size`: number of neurons in the output layer

**NOTE:** Before the call to build the network ( with `n3l_build()` ), if you don't have some custom function to initialize the weights or if you want to start learning from scratch without any previous state, is important to initialize the seed for the random function calling `srand()`.

At this point you only have to provide inputs to the `n3l_forward_propagation()` function. This function return the outputs of the neural network which you can use in your program.

To start the learning process, in other terms _backpropagate the outputs_, you have to provide the targets of your inputs and set the network outputs parameters with the results returned from the `n3l_forward_propagation` function. The new state will be passed as argument of `n3l_backward_propagation()`.

These last two points ( forward and backpropagation ) can be repeated many times, the important note here is to `free()` the network output each time to not spend too much memory.

When all the work is finish, you can free the neural network with `n3l_free()`.

Below a small example of XOR problem with N3 Library:

```C
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/* Include the header of n3 library */
#include <n3l/n3lib.h>

int main(int argc, char *argv[])
{
  N3LData *network;
  N3LArgs params;
  double inputs[4][2] = { { 0, 0}, { 0, 1 }, { 1, 1 }, {1, 0} };
  double targets[4][1] = { { 0 }, { 1 }, { 0 }, { 1 } };
  int iterations;

  /* Initializing parameters */
  params = n3l_get_default_args();
  params.bias = 0.5f;
  params.in_size = 2;
  params.out_size = 1;
  params.h_size = 3;
  params.h_layers = 1;

  /* Building network */
  srand(time(NULL));
  network = n3l_build(params, &n3l_rnd_weight);

  /* Learning mode */
  for ( iterations = 0; iterations < 10000; ++iterations ) {
    network->inputs = inputs[iterations % 4];
    network->outputs = n3l_forward_propagation(network);
    network->targets = targets[iterations % 4];
    n3l_backward_propagation(network);
  }

  /* Show results */
  for ( iterations = 0; iterations < 4; ++iterations ) {
    network->inputs = inputs[iterations],
    network->outputs = n3l_forward_propagation(network);
    printf("XOR - Case [ %.0lf, %.0lf ] - Result: %lf\n",
      network->inputs[0], network->inputs[1], network->outputs[0]);
  }

  /* Free the network */
  n3l_free(network);
}
```

Output sample:
```
XOR - Case [ 0, 0 ] - Result: 0.038544
XOR - Case [ 0, 1 ] - Result: 0.967957
XOR - Case [ 1, 1 ] - Result: 0.035713
XOR - Case [ 1, 0 ] - Result: 0.968785
```

If you want to know more about the parameters or for others functions like settings custom activation function, save the results, or load a previous one, see the documentation section.

## Documentation

### 1. Enumerators

#### 1.1 `bool`
Defines boolean values.

| Name      | Description |
|-----------|-------------|
| `false`   | Value 0     |
| `true`    | Value 1     |

#### 1.2 `N3LLayerType`
Defines the layer type. Values:

| Name             | Description               |
|------------------|---------------------------|
| `N3LInputLayer`  | _Input Layer ( value 0 )_ |
| `N3LHiddenLayer` | _Hidden Layer_            |
| `N3LOutputLayer` | _Output Layer_            |

#### 1.3 `N3LLogType`
Defines the log verbosity. Values:

| Name             | Description                                         |
|------------------|-----------------------------------------------------|
| `N3LLogNone`     | _No log at all ( value -1 )_                        |
| `N3LLogCritical` | _Log only the most important functions ( value 0 )_ |
| `N3LLogHigh`     | _Log the most important operations_                 |
| `N3LLogMedium`   | _Log important evaluation during the processes_     |
| `N3LLogLow`      | _Log weights initializations and delta results_     |
| `N3LLogPedantic` | _Log everything_                                    |

Each level include the log of the previous layers.

These values are used to log inside the library, if you set N3LLogNone, and this parameter is fixed by code, the same results can be obtained setting `NULL` the whole parameter `logger`.

See `N3LArgs` for more details.

#### 1.4 N3LActType
Defines the activation function. Values:

| Name         | Description                                         |
|--------------|-----------------------------------------------------|
| `N3LCustom`  | _Not an internal N3L function, you shouldn't use it directly. ( value -1 )_ |
| `N3LNone`    | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%20x) _( value 0)_ |
| `N3LSigmoid` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%201/%281%20+%20e%5E%7B-x%7D%20%29) |
| `N3LTanh`    | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%20tanh%28x%29) |
| `N3LRelu`    | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%20%5Cbegin%7Bmatrix%7D%20%5C%5C%20%26%20%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3C%200%20%5C%5C%20x%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cgeq%200%20%5Cend%7Bcases%7D%20%5Cend%7Bmatrix%7D) |

The use of `N3LCustom` is internal of N3 Library.

See `n3l_set_custom_act()` for more details.


### 2 Structures
**TODO**

### 3 Functions
**TODO**
