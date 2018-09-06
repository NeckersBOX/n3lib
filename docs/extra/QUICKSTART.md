# QuickStart

## Get and Build the library

```
$ git clone https://github.com/neckersbox/n3lib.git
$ cd n3lib
$ make
# make install
```

## Use the library

To use N3 Library in your project you need to follow the steps below:

1. Include the `n3l/n3lib.h` header in your source.
2. Build the network by `N3LArgs` and `n3l_network_build()`
3. Forward your inputs through the network by `n3l_forward_propagation()`
4. Improve network results providing targets and `n3l_backward_propagation()`
5. Free the memory used by the network with `n3l_network_free()`
6. Compile your project with the flag `-ln3l`

A small example implementing the XOR operation:
```C
#include <stdlib.h>
/** Include the library header **/
#include <n3l/n3lib.h>

int main(int argc, char *argv[])
{
  N3LArgs args;
  N3LNetwork *net;
  double inputs[4][2]  = { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 } };
  double targets[4][2] = { { 0 }, { 1 }, { 0 }, { 1 } };
  double *outputs;
  uint64_t j;

  /** Init with defaults parameters **/
  args = n3l_misc_init_arg();

  /** Define the number of neurons in the input layer **/
  args.in_size = 2;

  /** Build 1 hidden layer with 2 neurons using Sigmoid **/
  args.h_layers = 1;
  args.h_size = (uint64_t *) malloc(args.h_layers * sizeof(uint64_t));
  args.h_size[0] = 2;
  args.act_h = (N3LActType *) malloc(args.h_layers * sizeof(N3LActType));
  args.act_h[0] = N3LSigmoid;

  /** Define the number of neurons in the output layer **/
  args.out_size = 1;

  /** Setting bias **/
  args.bias = 0.5;

  /** Build the network with learning rate of 0.5 **/
  net = n3l_network_build(args, 0.5);

  /** Start learning with 10k iterations **/
  for ( j = 0; j < 10000; ++j ) {
    /** Forward Propagation **/
    net->inputs = inputs[j % 4];
    outputs = n3l_forward_propagation(net);
    /** At this point inputs and outputs could be free if needed **/

    printf("Iter %ld/%d - Target: %.0lf - Output: %lf - Error: %lf\n",
      j + 1, 10000, targets[j % 4][0], outputs[0], targets[j % 4][0] - outputs[0]);
    free(outputs);

    /** Backward Propagation **/
    net->targets = targets[j % 4];
    n3l_backward_propagation(net);
    /** At this point targets could be free if needed **/
  }

  /** Free the network state **/
  n3l_network_free(net);

  return 0;
}
```

Saved as `xor.c` can be compiled by `gcc xor.c -o xor -ln3l`, the results will be like this:

```
Iter 1/10000 - Target: 0 - Output: 0.678551 - Error: -0.678551
Iter 2/10000 - Target: 1 - Output: 0.703232 - Error: 0.296768
Iter 3/10000 - Target: 0 - Output: 0.730443 - Error: -0.730443
Iter 4/10000 - Target: 1 - Output: 0.676008 - Error: 0.323992
...
Iter 9997/10000 - Target: 0 - Output: 0.091856 - Error: -0.091856
Iter 9998/10000 - Target: 1 - Output: 0.930525 - Error: 0.069475
Iter 9999/10000 - Target: 0 - Output: 0.080475 - Error: -0.080475
Iter 10000/10000 - Target: 1 - Output: 0.930533 - Error: 0.069467
```

## Save and load results

You can also save the results and load them in another project using `n3l_file_export_network()` and `n3l_file_import_network()`.
Using the same example used in the previous section, before calling `n3l_network_free(net)` we can add the following line:

```C
if ( n3l_file_export_network(net, "xor.n3l") == true ) {
  printf("Network state saved to xor.n3l.\n");
}
else {
  fprintf(stderr, "Error while exporting state to xor.n3l\n");
}

/** Free the network state **/
n3l_network_free(net);
```

Now, if you want to load a previous network state without do again the learning process, you can load the file `xor.n3l`.
Example of another project which perform xor operation using the previous state saved:

```C
#include <stdlib.h>
#include <n3l/n3lib.h>

int main(int argc, char *argv[])
{
  N3LNetwork *net;
  double inputs[2][2] = { { 1, 0 }, { 1, 1 } };
  double *outputs;

  /** Load the network **/
  if ( !(net = n3l_file_import_network("xor.n3l")) ) {
    fprintf(stderr, "Error while loading xor.n3l\n");
    exit(1);
  }

  net->inputs = inputs[0];
  outputs = n3l_forward_propagation(net);
  printf("Case { 1, 0 } - Output: %lf\n", outputs[0]);
  free(outputs);

  net->inputs = inputs[1];
  outputs = n3l_forward_propagation(net);
  printf("Case { 1, 1 } - Output: %lf\n", outputs[0]);
  free(outputs);

  n3l_network_free(net);

  return 0;
}
```

And the results will be like these ones:

```
Case { 1, 0 } - Output: 0.931341
Case { 1, 1 } - Output: 0.079728
```

## Get inputs and targets from CSV

You can get inputs and targets data from a CSV file through the function `n3l_file_get_csv_data`.
Example using a file named `test.csv` with the following content:

```
Input 0, Input 1, Target 0
0.0,0.0,0.0
0.0,1.0,1.0
1.0,1.0,0.0
1.0,0.0,1.0
```

In this example the data are already in a _printf-like double style_. If you have inputs with space or others format the function accepts a custom parser function which receive the raw string and return
a double ( See `N3LCSVData` type )

```C
#include <stdlib.h>
/** Include the library header **/
#include <n3l/n3lib.h>

int main(int argc, char *argv[])
{
  N3LArgs args;
  N3LNetwork *net;
  double *outputs, *data;
  uint64_t j;
  FILE *csv;

  /** Open the CSV File **/
  if ( !(csv = fopen("test.csv", "r")) ) {
    fprintf(stderr, "Cannot open test.csv.\n");
    exit(1);
  }

  /** Init with defaults parameters **/
  args = n3l_misc_init_arg();

  /** Define the number of neurons in the input layer **/
  args.in_size = 2;

  /** Build 1 hidden layer with 2 neurons using Sigmoid **/
  args.h_layers = 1;
  args.h_size = (uint64_t *) malloc(args.h_layers * sizeof(uint64_t));
  args.h_size[0] = 2;
  args.act_h = (N3LActType *) malloc(args.h_layers * sizeof(N3LActType));
  args.act_h[0] = N3LSigmoid;

  /** Define the number of neurons in the output layer **/
  args.out_size = 1;

  /** Setting bias **/
  args.bias = 0.5;

  /** Build the network with learning rate of 0.5 **/
  net = n3l_network_build(args, 0.5);

  /** Start learning with 10k iterations **/
  net->inputs = (double *) malloc(2 * sizeof(double));
  net->targets = (double *) malloc(sizeof(double));
  for ( j = 0; j < 10000; ++j ) {
    /** Read the first 3 data and skip the first row */
    data = n3l_file_get_csv_data(csv, (j % 4) ? 0 : 1, 0, 3, NULL);

    /** Forward Propagation **/
    net->inputs[0] = data[0];
    net->inputs[1] = data[1];
    outputs = n3l_forward_propagation(net);

    /** Backward Propagation **/
    net->targets[0] = data[2];

    free(data);
    printf("Iter %ld/%d - Target: %.0lf - Output: %lf - Error: %lf\n",
      j + 1, 10000, net->targets[0], outputs[0], net->targets[0] - outputs[0]);
    free(outputs);

    n3l_backward_propagation(net);

    /** If this row is the last of the file, set the file cursor
        again to the start position **/
    if ( !((j + 1) % 4) ) {
      printf("End example. Restart.\n");
      rewind(csv);
    }
  }
  free(net->inputs);
  free(net->targets);
  fclose(csv);

  /** Free the network state **/
  n3l_network_free(net);

  return 0;
}
```
