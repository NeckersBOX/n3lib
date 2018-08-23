![Build](https://img.shields.io/badge/build-passing-green.svg) ![Status](https://img.shields.io/badge/status-dev-orange.svg) ![Version](https://img.shields.io/badge/version-1.4.0-lightgray.svg)

# N3 Library
A tiny C library for building neural network with the capability to define custom activation functions, learning rate, bias and others parameters.

Both forward and backpropagation are built to parallelize through threads the operations.

In forward propagation each layer execute the _activation functions_ in concurrent threads, when one layer end its own job, the results to the each next layer's neuron are collected with others concurrent threads. The whole process is executed from each layer from the input one to the output one.

In backward propagation first there is the delta evaluation from the output layer to the input layer, after that there is the weight updating process from the input to the output. Each return to the previous layer and delta evaluation on neurons in the same layer, is built, even in this case, with concurrent threads.

Due to the list type structure of layers and neurons, N3 Library allow you to remove and add neurons or layers even while it's in running mode ( at the end of each iteration ).

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

You can disable the internal N3 Library log with the flag `-DN3L_DISABLE_LOG`. Example:
```
make flags="-DN3L_DISABLE_LOG"
```
