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

This last two points ( forward and backpropagation ) can be repeated many times, the important note here is to `free()` the network output each time to not spend too much memory.

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
Yeah, it's a good question.. Coming soon
