![Build](https://img.shields.io/badge/build-passing-green.svg) ![Status](https://img.shields.io/badge/status-alpha-yellow.svg) ![Version](https://img.shields.io/badge/version-1.3.0-lightgray.svg)
# N3 Library
A tiny C library for building neural network with the capability to define custom activation functions, learning rate, bias and others parameters.

Both forward and backpropagation are built to parallelize through threads the operations.

In forward propagation each layer execute the _activation functions_ in concurrent threads, when one layer end its own job, the results to the each next layer's neuron are collected with others concurrent threads. The whole process is executed from each layer from the input one to the output one.

In backward propagation first there is the delta evaluation from the output layer to the input layer, after that there is the weight updating process from the input to the output. Each return to the previous layer and delta evaluation on neurons in the same layer, is built, even in this case, with concurrent threads.

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

Build a network with N3 Library without many options is fast. First you need to initialize the parameters through a `N3LArgs` structure.
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
// Include the header of n3 library
#include <n3l/n3lib.h>

int main(int argc, char *argv[])
{
  N3LData * network;
  N3LArgs params;
  double inputs[4][2] = { { 0, 0}, { 0, 1 }, { 1, 1 }, {1, 0} };
  double targets[4][1] = { { 0 }, { 1 }, { 0 }, { 1 } };
  int iterations;

  // Initializing parameters
  params = n3l_get_default_args();
  params.bias = 0.5f;
  params.in_size = 2;
  params.out_size = 1;
  params.h_size = 3;
  params.h_layers = 1;

  // Building network
  srand(time(NULL));
  network = n3l_build(params, &n3l_rnd_weight);

  // Learning mode
  for ( iterations = 0; iterations < 10000; ++iterations ) {
    network->inputs = inputs[iterations % 4];
    network->outputs = n3l_forward_propagation(network);
    network->targets = targets[iterations % 4];
    n3l_backward_propagation(network);
  }

  // Show results
  for ( iterations = 0; iterations < 4; ++iterations ) {
    network->inputs = inputs[iterations],
    network->outputs = n3l_forward_propagation(network);
    printf("XOR - Case [ %.0lf, %.0lf ] - Result: %lf\n",
      network->inputs[0], network->inputs[1], network->outputs[0]);
  }

  // Free the network
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

### 1. Macro and Define

#### 1.1 N3L_VERSION
Current N3 Library version as string, i.e. "1.2.9"

#### 1.2 N3L_ACT(fun)

Pointer to functions of type:
```c
double fun(double value)
```

Used in neurons as activation function. Argument `value` is the neuron's input.

See also `N3LNeuron`, `N3LActType` or activation functions such as `n3l_sigmoid()`, `n3l_sigmoid_prime`, ...

#### 1.3 N3L_RND_WEIGHT(rnd_w)

Pointer to functions of type:
```c
double rnd_w(N3LLayer layer)
```

Used during weights initialization to get random values. Argument `layer` is the current layer to initialize.

See also `N3LData`, `N3LLayer`, `n3l_rnd_weight()`.

### 2. Enumerators

#### 2.1 `bool`
Defines boolean values.

| Name      | Description |
|-----------|-------------|
| `false`   | Value 0     |
| `true`    | Value 1     |

#### 2.2 `N3LLayerType`
Defines the layer type. Values:

| Name             | Description               |
|------------------|---------------------------|
| `N3LInputLayer`  | _Input Layer ( value 0 )_ |
| `N3LHiddenLayer` | _Hidden Layer_            |
| `N3LOutputLayer` | _Output Layer_            |

#### 2.3 `N3LLogType`
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

#### 2.4 N3LActType
Defines the activation function. Values:

| Name         | Description                                         |
|--------------|-----------------------------------------------------|
| `N3LCustom`  | _Not an internal N3L function, you shouldn't use it directly. ( value -1 )_ |
| `N3LNone`    | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%20x) _( value 0)_ |
| `N3LSigmoid` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;%20e%5E%7B-x%7D%7D) |
| `N3LTanh`    | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%20tanh%28x%29) |
| `N3LRelu`    | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7B%20if%20%7D%20x%20%3C%200%20%5C%5C%20x%26%20%5Ctext%7B%20if%20%7D%20x%20%5Cgeq%200%20%5Cend%7Bcases%7D) |
| `N3LIdentity` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%3D%20x) |
| `N3LLeakyRelu` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cbegin%7Bcases%7D%200.01x%20%26%20%5Ctext%7Bfor%20%7D%20x%20%3C%200%5C%5C%20x%20%26%20%5Ctext%7Bfor%20%7D%20x%20%5Cge%200%5Cend%7Bcases%7D) |
| `N3LSoftPlus` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cln%281&plus;e%5E%7Bx%7D%29) |
| `N3LSoftSign` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cfrac%7Bx%7D%7B1%20&plus;%20%7Cx%7C%7D) |
| `N3LSwish` | ![eq](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20x%20*%20%5Csigma%28x%29) |

The use of `N3LCustom` is internal of N3 Library.

See `n3l_set_custom_act()` for more details.

Reference: [Wikipedia - Activation Function](https://en.wikipedia.org/wiki/Activation_function)

### 3 Structures

#### 3.1 N3LLogger

| Type | Name | Description |
|------|------|-------------|
|`FILE *` | `log_file` | Pointer to an already opened file in _read_ mode. |
|`N3LLogType` | `verbosity` | Threshold level to write the log |

#### 3.2 N3LNeuron

| Type | Name | Description |
|------|------|-------------|
| `double` | `input` | Neuron input
| `double *` | `weights` | Weights array with length equal to the `outputs` term. |
| `uint64_t` | `outputs` | Number of outputs linked to this neuron |
| `double` | `result` | Value returned from the activation function |
| `N3L_ACT`| `act` | Activation function |
| `N3L_ACT`| `act_prime` | Derivate of the activation function |

#### 3.3 N3LLayer

| Type | Name | Description |
|------|------|-------------|
| `N3LLayerType` | `ltype` | Type of this layer |
| `uint64_t` | `size` | Number of neurons in this layer |
| `N3LNeuron *` | `neurons` | Neurons array with length equal of `size` term. |

#### 3.4 N3LArgs

| Type | Name | Description |
|------|------|-------------|
| `bool` | `read_file` | Set `true` if during initialization the values will be read from the file `in_filename`. |
| `char *` | `in_filename` | Dependent from `read_file`. Path and filename which contains the values to initialize the network. |
| `double` | `bias` | Bias term value. If you don't want any bias term set this value to zero. High values could saturate the network. |
| `double` | `learning_rate` | Learning rate value. Higher value could lead to instability during backpropagation process. |
| `uint64_t` | `in_size` | Number of input neurons. |
| `uint64_t` | `h_size` | Number of hidden neurons for each hidden layer. |
| `uint64_t` | `out_size` | Number of output neurons. |
| `uint64_t` | `h_layers` | Number of hidden layers. |
| `N3LLogger *` | `logger` | Pointer to a logger used from the library's function. If you don't want any log, it can be set to `NULL` |
| `N3LActType` | `act_in` | Type of activation function used by neurons in the input layer. |
| `N3LActType` | `act_h` | Type of activation function used by neurons in the hidden layer. |
| `N3LActType` | `act_out` | Type of activation function used by neurons in the output layer. |

#### 3.5 N3LData

| Type | Name | Description |
|------|------|-------------|
|`double *`|`inputs`| Array of inputs values used in both backward and forward propagation. |
|`double *`|`targets`| Array of target values used in backpropagation. These are the _ideal_ output value to reach for the inputs values provided. |
|`double *`|`outputs`| Array of outputs values used in backpropagation. Usually this should be set with the return value of `n3l_forward_propagation()` and manually free. |
| `N3L_RND_WEIGHT` | `get_rnd_weight` | Function to call when weights are initialized randomly. |
| `N3LArgs *` | `args` | Network parameters. |
| `N3LLayer *` | `net` | Network layers array |

### 4 Functions

#### 4.1 Activation
This functions are used internally through the `N3LActType` which information is saved along with the others network data. If you use them manually the information could not be stored in the file ( the value will be `N3LCustom` ) and you have to set again after the initialization process with `n3l_set_custom_act()`.

##### 4.1.1 `n3l_act_none()`

```c
double n3l_act_none(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?none%28value%29%3D%20value) |

##### 4.1.2 `n3l_act_relu()`

```c
double n3l_act_relu(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?relu%28value%29%20%3D%20%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7B%20if%20%7D%20value%20%3C%200%20%5C%5C%20value%26%20%5Ctext%7B%20if%20%7D%20value%20%5Cgeq%200%20%5Cend%7Bcases%7D) |

##### 4.1.3 `n3l_act_relu_prime()`

```c
double n3l_act_relu_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f'%28value%29%3D%20%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7B%20if%20%7D%20value%20%5Cleq%200%20%5C%5C%201%26%20%5Ctext%7B%20if%20%7D%20value%20%3E%200%20%5Cend%7Bcases%7D) |

##### 4.1.4 `n3l_act_sigmoid()`

```c
double n3l_act_sigmoid(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?sigmoid%28value%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;%20e%5E%7B-value%7D%7D) |

##### 4.1.5 `n3l_act_sigmoid_prime()`

```c
double n3l_act_sigmoid_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f'%28value%29%3D%20sigmoid%28value%29%20*%20%281%20-%20sigmoid%28value%29%29) |

##### 4.1.6 `n3l_act_tanh()`

```c
double n3l_act_tanh(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?tanh%28value%29%3D%20%5Cfrac%7Be%5E%7Bvalue%7D-e%5E%7B-value%7D%7D%7Be%5E%7Bvalue%7D+e%5E%7B-value%7D%7D) |

##### 4.1.7 `n3l_act_tanh_prime()`

```c
double n3l_act_tanh_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f'%28value%29%20%3D%201%20-%20tanh%28value%29%5E%7B2%7D) |

##### 4.1.8 `n3l_act_identity()`

```c
double n3l_act_identity(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?identity%28value%29%20%3D%20value) |

##### 4.1.9 `n3l_act_identity_prime()`

```c
double n3l_act_identity_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f%27%28value%29%3D1) |

##### 4.1.10 `n3l_act_leaky_relu()`

```c
double n3l_act_leaky_relu(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?leaky\_relu%28value%29%20%3D%20%5Cbegin%7Bcases%7D%200.01value%20%26%20%5Ctext%7Bfor%20%7D%20value%20%3C%200%5C%5C%20value%20%26%20%5Ctext%7Bfor%20%7D%20value%20%5Cge%200%5Cend%7Bcases%7D) |

##### 4.1.11 `n3l_act_leaky_relu_prime()`

```c
double n3l_act_leaky_relu_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f%27%28value%29%20%3D%20%5Cbegin%7Bcases%7D%200.01%20%26%20%5Ctext%7Bfor%20%7D%20value%20%3C%200%5C%5C%201%20%26%20%5Ctext%7Bfor%20%7D%20value%20%5Cge%200%5Cend%7Bcases%7D) |

##### 4.1.12 `n3l_act_softplus()`

```c
double n3l_act_softplus(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?softplus%28value%29%20%3D%20%5Cln%281%20&plus;%20e%5E%7Bvalue%7D%29) |

##### 4.1.13 `n3l_act_softplus_prime()`

```c
double n3l_act_softplus_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f%27%28value%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;%20e%5E%7B-value%7D%7D) |

##### 4.1.14 `n3l_act_softsign()`

```c
double n3l_act_softsign(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?softsign%28value%29%3D%5Cfrac%7Bvalue%7D%7B1&plus;%7Cvalue%7C%7D) |

##### 4.1.15 `n3l_act_softsign_prime()`

```c
double n3l_act_softsign_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f%27%28value%29%3D%5Cfrac%7B1%7D%7B%281&plus;%7Cvalue%7C%29%5E2%7D) |

##### 4.1.16 `n3l_act_swish()`

```c
double n3l_act_swish(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?swish%28value%29%3Dvalue%20*%20sigmoid%28value%29) |

##### 4.1.17 `n3l_act_swish_prime()`

```c
double n3l_act_swish_prime(double value)
```

| Formula |
|---------|
| ![eq](https://latex.codecogs.com/gif.latex?f%27%28value%29%3Dswish%28value%29%20&plus;%20sigmoid%28value%29%20*%20%281%20-%20swish%28value%29%29) |

#### 4.2 Backward Propagation

##### 4.2.1 `n3l_backward_propagation()`

```c
void n3l_backward_propagation(N3LData *net)
```

Argument `net` is the current initialized network. This function adjusts the weights according to the errors results from the difference between `net->targets` and `net->outputs`.

#### 4.3 Forward Propagation

##### 4.3.1 `n3l_forward_propagation()`

```c
double *n3l_forward_propagation(N3LData *net)
```

Argument `net` is the current initialized network. This function get the outputs from the `net->inputs` provided.

Return an array with length `net->args->out_size` with the results obtained. This results should be set to `net->outputs` if you are in learning mode and must be free manually.

#### 4.4 Initialization

##### 4.4.1 `n3l_build()`

```c
N3LData *n3l_build (N3LArgs args, N3L_RND_WEIGHT(rnd_w))
```

Initialize the values of the whole network and return the network built.

If you don't want to set all the `args`, you can call `n3l_get_default_args()` and set only the arguments needed from your project.

As argument `rnd_w`, if you don't bother about weights random initialization or you have set a file to read during initialization, can be set to `&n3l_rnd_weight`.

**Note:** Before call this function, to initialize the seed for `rand()`, you have to call `srand()` manually.

##### 4.4.2 `n3l_get_default_args()`

```c
N3LArgs n3l_get_default_args(void)
```

Set defaults values for the neural network.

| Variable        | Default Value |
|-----------------|---------------|
| `read_file`     | `false`       |
| `in_filename`   | `NULL`        |
| `bias`          | `0`           |
| `learning_rate` | `1`           |
| `in_size`       | `0`           |
| `h_size`        | `0`           |
| `h_layers`      | `0`           |
| `out_size`      | `0`           |
| `logger`        | `NULL`        |
| `act_in`        | `N3LNone`     |
| `act_h`         | `N3LSigmoid`  |
| `act_out`       | `N3LSigmoid`  |


##### 4.4.3 `n3l_rnd_weight()`

```c
double n3l_rnd_weight (N3LLayer layer)
```

Generate a random value to use in weights initialization.

Return a value in range `[0, 1]` using `rand()` function.

##### 4.4.5 `n3l_set_custom_act()`

```c
void n3l_set_custom_act (N3LData *net, uint64_t layer_index, N3L_ACT(act), N3L_ACT(act_prime))
```

Set a custom activation function, and its derivate, to neurons of layer `layer_index`.

When applied the information in `act_in`, `act_h` or `act_out` ( depends by layer type at index provided ) is set to `N3LCustom`.

#### 4.5 Logger
The log is write only if the parameter `logger` is initialized and its `type` argument is not equal to `N3LLogNone` and less or equal of `verbosity`.

A log example:
```
[N3Lib] [2] [n3l_build_bias] -->>
[N3Lib] [2] [n3l_build_bias] Building bias neuron (0,2)
[N3Lib] [2] [n3l_build_bias] <<--
```

The format is the following:

[`Module`] [`type`] [`fun_name`] [`message`]

##### 4.5.1 `n3l_log_start()`

```c
void n3l_log_start(N3LLogger *logger, const char *fun_name, N3LLogType type)
```

Print the log with message like `-->>`.

##### 4.5.2 `n3l_log()`

```c
void n3l_log (N3LLogger *logger, const char *fun_name, N3LLogType type, const char *message, ...)
```

Print the log with message `message` resolving its arguments ( in _printf_ mode ) with the optional next parameters.

**NOTE:** The print is in sync mode due to threads use. Before to print the text resolve the entire message in a single string with max length of 8192 characters. To finalize the print there is an explicit call to the `fflush()` function.

##### 4.5.3 `n3l_log_end()`

```c
void n3l_log_end (N3LLogger *logger, const char *fun_name, N3LLogType type)
```

Print the log with message like `<<--`.

#### 4.6 Neural Network Utilities

##### 4.6.1 `n3l_free()`

```c
void n3l_free (N3LData *net)
```

Free all the the memory allocate from `net`.

##### 4.6.2 `n3l_clone()`

```c
N3LData *n3l_clone (N3LData *net)
```

Clone the whole network `net` in a new one.

#### 4.7 Saving


##### 4.7.1 `n3l_save()`

```c
void n3l_save (N3LData *net, FILE *of)
```

Save the whole network to the, already opened in write mode, file `of`.

See also `N3 Library File Format`.


##  N3 Library File Format

The structure of the file saved or read is the following:

| Type | Description |
|------|-------------|
| `uint64_t`   | Input neurons number  |
| `uint64_t`   | Hidden neurons number |
| `uint64_t`   | Output neurons number |
| `uint64_t`   | Hidden layers number  |
| `double`     | Bias value            |
| `N3LActType` | Input layer activation type  |
| `N3LActType` | Hidden layer activation type |
| `N3LActType` | Output layer activation type |

Next for each layer in the network ( from 0 ) write in order for each neurons ( from 0 )

| Type | Description |
|------|-------------|
| `double [size equal to current neuron outputs]` | Weight values for the current neuron  |
