**GitHub Project:** [https://github.com/NeckersBOX/n3lib](https://github.com/NeckersBOX/n3lib)

A C library for building neural network with the capability to define custom activation functions, learning rate, bias and others parameters.

Both forward and backpropagation are built to parallelize through threads the operations.

In forward propagation each layer execute the _activation functions_ in concurrent threads, when one layer end its own job, the results to the each next layer's neuron are collected with others concurrent threads. The whole process is executed from each layer from the input one to the output one.

In backward propagation first there is the delta evaluation from the output layer to the input layer, after that there is the weight updating process from the input to the output. Each return to the previous layer and delta evaluation on neurons in the same layer, is built, even in this case, with concurrent threads.

Due to the list type structure of layers and neurons, N3 Library allow you to remove and add neurons or layers even while it's in running mode ( at the end of each iteration ).

## Build
The library is designed to run under GNU\\Linux operating systems.

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

## Documentation

#### Online Docs: [https://neckersbox.github.io/n3lib/](https://neckersbox.github.io/n3lib/)

The library is written to use Doxygen to generate up-to-date documentation and man pages.
You can update the docs executing the following command:

```
doxygen n3lib.doxygen.conf
```

And install man pages by:

```
# make install doc
```

Pre generated documentation is provided along with the library in the folder `docs`.
More content, not strictly linked with the code, are included in the path `docs\extra`.

## Running Examples

There are small projects built with N3 Library into the path `examples`, along with sources you can found report with details about memory, execution times and performance graphs.

**NOTE:** File with extension `.n3l` contains the network state after learning with number of iterations equals to the same written in relative reports.

## License
The library is released under BSD-2-Clause License.

You can found a copy of the license into the `LICENSE` file.
